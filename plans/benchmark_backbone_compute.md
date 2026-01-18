# Benchmark: Image Backbone vs Transformer Compute

## Context

We're considering an optimization to the RTC (Real-Time Chunking) training in `ACTRelativeRTCPolicy`. Currently, training samples one random delay per sample. The VFLASH paper shows that evaluating all delays in a single forward pass improves training efficiency.

Two implementation approaches exist:
1. **VFLASH-style**: Custom attention masks, complex implementation
2. **Batch expansion**: Expand batch after backbone, simple implementation

Option 2's viability depends on the assumption that **the image backbone is 80-90% of forward pass compute**. If true, expanding the batch for transformer layers (but not backbone) yields most of the efficiency gain with minimal code changes.

## Goal

Benchmark the `ACTRelativeRTC` model to measure the proportion of forward pass time spent in:
1. Image backbone (ResNet feature extraction)
2. Encoder transformer layers
3. Decoder transformer layers
4. Other operations (projections, VAE encoder if enabled)

## Implementation

Create a script at `packages/lerobot_policy_act_relative_rtc_2/scripts/benchmark_forward_pass.py`.

### Setup

```python
import torch
import time
from lerobot_policy_act_relative_rtc_2.configuration_act_relative_rtc import ACTRelativeRTCConfig
from lerobot_policy_act_relative_rtc_2.modeling_act_relative_rtc import ACTRelativeRTCPolicy

# Use realistic config matching our training setup
config = ACTRelativeRTCConfig(
    chunk_size=50,
    rtc_max_delay=10,
    n_obs_steps=1,
    # Image config (2 cameras, 480x640 typical for SO-101)
    image_features={
        "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"},
        "observation.images.wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    },
    vision_backbone="resnet18",  # or resnet50 if that's what we use
    # State/action dims for SO-101
    state_feature={"shape": [6], "type": "STATE"},
    action_feature={"shape": [6], "type": "ACTION"},
    # Transformer config
    dim_model=256,
    n_encoder_layers=4,
    n_decoder_layers=1,
    n_heads=8,
)
```

### Benchmark Method

Use `torch.cuda.Event` for accurate GPU timing (not `time.time()`):

```python
def benchmark_component(fn, warmup=10, iterations=100):
    """Benchmark a function with proper CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations  # ms per iteration
```

### Components to Benchmark

Modify or wrap the model to isolate each component:

1. **Full forward pass**: `model.forward(batch)`
2. **Backbone only**: `model.model.backbone(images)`
3. **Encoder only**: Given pre-computed image features, run encoder
4. **Decoder only**: Given encoder output, run decoder

Create synthetic batch data:
```python
batch_size = 32
device = "cuda"

batch = {
    "observation.images.top": torch.randn(batch_size, 3, 480, 640, device=device),
    "observation.images.wrist": torch.randn(batch_size, 3, 480, 640, device=device),
    "observation.state": torch.randn(batch_size, 6, device=device),
    "action": torch.randn(batch_size, 60, 6, device=device),  # max_delay + chunk_size
    "action_is_pad": torch.zeros(batch_size, 60, dtype=torch.bool, device=device),
}
```

### Expected Output

Print a table like:
```
Component               Time (ms)    % of Forward
--------------------------------------------------
Full forward            XX.XX        100.0%
  Backbone (ResNet)     XX.XX        XX.X%
  Image projection      XX.XX        XX.X%
  Encoder               XX.XX        XX.X%
  Decoder               XX.XX        XX.X%
  Other                 XX.XX        XX.X%
```

### Variations to Test

1. **Batch sizes**: 8, 16, 32, 64 (if memory allows)
2. **Image resolutions**: 240x320, 480x640 (to see scaling)
3. **Number of cameras**: 1, 2
4. **Backbone**: resnet18 vs resnet50

## Success Criteria

- If backbone ≥ 70% of forward pass: Option B (batch expansion) is viable
- If backbone < 50% of forward pass: Need to reconsider, VFLASH-style may be necessary

## Notes

- Run on the same GPU we use for training (check with `nvidia-smi`)
- Ensure model is in training mode (`model.train()`) since that's what we're optimizing
- Run multiple times to ensure consistency
- Consider memory usage too: report peak GPU memory for each batch size
