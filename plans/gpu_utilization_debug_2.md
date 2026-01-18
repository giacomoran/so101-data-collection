# GPU Utilization Debugging Plan (act_relative_rtc_2)

## Goal

Measure and optimize GPU utilization for `act_relative_rtc_2` training. Compare against standard `act` policy baseline.

**Baseline Results** (to be confirmed):
- Standard ACT: Expected good GPU utilization: 75%
- act_relative_rtc_2: 55%

## Policy Architecture (V2)

The `act_relative_rtc_2` policy is a V2 redesign that extends standard ACT with:

1. **Relative action representation**: Actions are predicted relative to current observation state
   - `relative_action = action - obs.state[t]`
   - Only relative actions are normalized (via `relative_action_normalizer`)

2. **Simplified observation handling** (fixed from V1):
   - `observation_delta_indices = [0]` - loads single observation frame
   - No double image loading (V1 bug fixed)
   - No delta_obs computation or normalization
   - No observation queue for inference

3. **Simplified encoder architecture** (V2 changes):
   - Removed robot state input projection from encoder
   - Encoder input: latent token + env_state (if present) + image features
   - No robot_state tokens in VAE encoder either

4. **RTC (Real-Time Chunking)** for action prefix conditioning:
   - Training evaluates all delays {0, ..., rtc_max_delay} in parallel
   - Expands batch from B to B*D virtual samples after backbone
   - Action indices: `[1, ..., rtc_max_delay + chunk_size]` (skips action[0])

5. **Normalization strategy**:
   - STATE/ACTION: `IDENTITY` (no normalization - computed on absolute values before relative transform)
   - VISUAL: `MEAN_STD` (standard ImageNet normalization)

The preprocessed dataset `so101_data_collection_cube_hand_guided_1x224x8` already has images resized to 224x224.

## Your Task

As a coding agent, you will:
1. Read the progress file at `outputs/tmp/gpu_debug_progress_2.md`
2. If baselines are missing, record them from the completed runs
3. If baselines exist, come up with an hypothesis which has not been tested already
4. Implement and test the hypothesis
5. Update the progress file with results
6. Stop (next agent will continue)

Note that we already ran some tests previously on pretty much the same model, you can see the results at `outputs/tmp/gpu_debug_progress.md`.

## How to Run Checks

### Test a Hypothesis

Run the act_relative_rtc_2 training with your modifications using the GPU monitoring script:

```bash
python packages/so101_data_collection/src/so101_data_collection/train/debug_gpu_utilization_2.py --output outputs/tmp/gpu_test_hypothesis_N.csv --cmd 'lerobot-train \
    --policy.type=act_relative_rtc_2 \
    --dataset.repo_id=giacomoran/so101_data_collection_cube_hand_guided_1x224x8 \
    --dataset.episodes=[0] \
    --steps=200 \
    --batch_size=8 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=8 \
    --policy.n_action_steps=8 \
    --policy.obs_state_delta_frames=1 \
    --policy.rtc_max_delay=3 \
    --policy.use_vae=false \
    --policy.n_decoder_layers=4 \
    --policy.downscale_img_square=224 \
    --policy.vision_backbone=resnet34 \
    --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 \
    --policy.pre_norm=true \
    --policy.device=cuda \
    --policy.input_features='"'"'{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 224, 224], "type": "VISUAL"}}'"'"' \
    --wandb.enable=false \
    --output_dir=outputs/tmp/test_hypothesis_N \
    --policy.push_to_hub=false \
    [ADD YOUR MODIFIED FLAGS HERE]'
```

Replace `hypothesis_N` with your hypothesis number. Add or modify flags as needed for your test.

### Verify Results

The script will print GPU summary statistics. Compare:
- **Baseline act_relative_rtc_2**: Check progress file for the baseline GPU util
- **Your test**: GPU util from your run
- **Improvement**: Calculate the difference

**Note**: The monitoring script waits for GPU usage to reach 20% before starting measurements, ensuring the initialization phase is excluded from statistics.

## Success Criteria

A hypothesis **succeeds** when:
- GPU utilization improves by **≥20 percentage points** vs baseline act_relative_rtc_2
- Training completes without errors
- Loss decreases over training steps (not NaN/diverging)

A hypothesis **fails** when:
- GPU utilization improves by <10 percentage points
- Training crashes or produces errors
- Loss is NaN or diverges

Note: Loss values are not comparable between ACT and act_relative_rtc_2 due to different architectures.

## Progress File Format

Update `outputs/tmp/gpu_debug_progress_2.md` after testing:

```markdown
### Hypothesis N: [Title]
- **Description**: [What you're testing]
- **Test Method**: [What you changed/measured]
- **Status**: VALIDATED | REJECTED
- **Result**:
  - GPU util before: [baseline]%
  - GPU util after: [your result]%
  - Improvement: [delta]pp
  - Training: [SUCCESS/FAILED - errors?]
  - Loss: [CONVERGING/NaN/DIVERGING]
  - Notes: [Any observations, side effects, or insights]
```

## Code Modification Guidelines

When implementing fixes:

1. **Make minimal changes**: Test one variable at a time
2. **Preserve correctness**: Don't break the policy's relative transformation logic
3. **Document changes**: Add comments explaining why modifications were made
4. **Compare carefully**: Ensure loss still converges (different absolute values are OK)

## Critical Files Reference

### Policy Implementation
- `packages/lerobot_policy_act_relative_rtc_2/src/lerobot_policy_act_relative_rtc_2/`
  - `configuration_act_relative_rtc.py` - Config and delta_timestamps logic
  - `modeling_act_relative_rtc.py` - Policy forward pass
  - `processor_act_relative_rtc.py` - Preprocessing pipeline
  - `relative_stats.py` - Statistics computation

### Dataset
- Dataset: `giacomoran/so101_data_collection_cube_hand_guided_1x224x8`
- Location: `~/.cache/huggingface/hub/datasets--giacomoran--so101_data_collection_cube_hand_guided_1x224x8/`
- Images: Already resized to 224x224
- Relative stats: Should be in `meta/relative_stats.json`

### LeRobot Training
- `.venv/lib/python3.10/site-packages/lerobot/scripts/lerobot_train.py` - Main training script
- DataLoader config at lines 340-349

## After Your Hypothesis

1. Update `outputs/tmp/gpu_debug_progress_2.md` with your results
2. If hypothesis succeeded: Document the fix and suggest next steps
3. If hypothesis failed: Document why and move to next hypothesis
4. **Stop** - Do not test multiple hypotheses in one session

The next coding agent will read the progress file and continue with the next hypothesis.
