# Dataset Preprocessing Tool for ACT Relative RTC

## Goal

Create a preprocessing tool that prepares LeRobot datasets for ACT Relative RTC training by:

1. Precomputing relative stats (delta_obs, relative_action) and storing in `meta/relative_stats.json`
2. Preprocessing images to target resolution (pad to square + resize)

This eliminates runtime overhead from:

- ~2min relative stats computation at every training init
- GPU utilization issues from runtime image resizing

## Design Decisions

### Relative Stats Storage: `meta/relative_stats.json` (separate file)

- Clean separation from LeRobot's standard `stats.json`
- Easy detection: file exists = stats precomputed
- Won't confuse LeRobot's default normalizers

### Image Preprocessing: Create new dataset copy

- Preserves original dataset
- User specifies output dataset name via `--output-repo-id`
- Videos re-encoded at target resolution using ffmpeg directly (not frame-by-frame)

### Video Resizing: ffmpeg filter_complex

Use ffmpeg's `pad` and `scale` filters directly on video files:

```bash
ffmpeg -i input.mp4 -vf "pad=max(iw\,ih):max(iw\,ih):(ow-iw)/2:(oh-ih)/2:black,scale=224:224" output.mp4
```

This is much faster than decoding frames, processing in Python, and re-encoding.

## Files to Create/Modify

### 1. NEW: `packages/so101_data_collection/src/so101_data_collection/collect/preprocess_dataset.py`

Main preprocessing tool with CLI interface:

```bash
# Full preprocessing
python -m so101_data_collection.collect.preprocess_dataset \
    --repo-id giacomoran/so101_data_collection_cube_hand_guided \
    --output-repo-id giacomoran/cube_hand_guided_224 \
    --target-resolution 224

# Relative stats only (in-place, adds meta/relative_stats.json)
python -m so101_data_collection.collect.preprocess_dataset \
    --repo-id giacomoran/so101_data_collection_cube_hand_guided \
    --relative-stats-only

# Push to HuggingFace Hub
python -m so101_data_collection.collect.preprocess_dataset \
    --repo-id giacomoran/so101_data_collection_cube_hand_guided \
    --output-repo-id giacomoran/cube_hand_guided_224 \
    --target-resolution 224 \
    --push-to-hub
```

Key functions:

- `compute_and_save_relative_stats()`: Compute stats and write to `meta/relative_stats.json`
- `resize_video_ffmpeg()`: Use ffmpeg to pad+resize video directly
- `preprocess_dataset()`: Main orchestration

### 2. MODIFY: `modeling_act_relative_rtc.py`

Add method to load precomputed stats:

```python
def _try_load_precomputed_relative_stats(self, dataset_meta) -> bool:
    """Try to load relative stats from meta/relative_stats.json."""
    relative_stats_path = Path(dataset_meta.root) / "meta" / "relative_stats.json"
    if not relative_stats_path.exists():
        return False
    # Load and configure normalizers
    ...
```

Modify `__init__()` to try loading precomputed stats first.

### 3. MODIFY: `processor_act_relative_rtc.py`

Add early-exit for already-preprocessed images:

```python
def _is_preprocessed(self, image: torch.Tensor) -> bool:
    """Check if image is already at target resolution."""
    h, w = image.shape[-2:]
    return h == self.downscale_img_square and w == self.downscale_img_square
```

## `meta/relative_stats.json` Format

```json
{
  "delta_obs": {
    "mean": [0.0001, -0.0002, 0.0003, 0.0001, 0.0, 0.0002],
    "std": [0.0234, 0.0189, 0.0312, 0.0156, 0.0098, 0.0445]
  },
  "relative_action": {
    "mean": [0.0012, 0.0034, 0.0056, 0.0023, 0.0011, 0.0089],
    "std": [0.1234, 0.0987, 0.1456, 0.0765, 0.0543, 0.2345]
  },
  "config": {
    "obs_state_delta_frames": 1,
    "chunk_size": 100,
    "total_samples": 12345,
    "created_at": "2025-01-13T12:00:00Z"
  }
}
```

## Implementation Steps

1. **Create `preprocess_dataset.py`**

   - Argparse CLI with options: `--repo-id`, `--output-repo-id`, `--target-resolution`, `--relative-stats-only`, `--push-to-hub`
   - Reuse `compute_relative_stats()` from `relative_stats.py`
   - Use ffmpeg subprocess for video resizing (pad to square + scale)

2. **Modify policy to load precomputed stats**

   - Add `_try_load_precomputed_relative_stats()` method
   - Update `__init__()` to check for precomputed stats first

3. **Modify processor to skip preprocessing**
   - Add `_is_preprocessed()` check
   - Return early if images already at target resolution

## Verification

1. **Test relative stats computation**:

   ```bash
   python -m so101_data_collection.collect.preprocess_dataset \
       --repo-id giacomoran/so101_data_collection_cube_hand_guided \
       --relative-stats-only
   # Verify meta/relative_stats.json created
   ```

2. **Test full preprocessing**:

   ```bash
   python -m so101_data_collection.collect.preprocess_dataset \
       --repo-id giacomoran/so101_data_collection_cube_hand_guided \
       --output-repo-id giacomoran/cube_hand_guided_224 \
       --target-resolution 224
   # Verify new dataset created with resized videos
   ```

3. **Test training with preprocessed dataset**:
   ```bash
   lerobot-train \
       --dataset.repo_id giacomoran/cube_hand_guided_224 \
       --policy.type act_relative_rtc
   # Verify: No "Computing relative stats" log message
   # Verify: Lower GPU memory usage from skipped image resize
   ```
