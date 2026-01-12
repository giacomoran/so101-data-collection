# ACT Relative RTC Policy

A modified ACT (Action Chunking Transformer) policy that uses relative joint representations, designed for improved generalization in robot manipulation tasks.

## Overview

This policy differs from standard ACT in two key ways:

1. **Relative Actions**: Actions are predicted relative to the current observation state (`action - obs.state[t]`) rather than as absolute joint positions
2. **Observation Deltas**: The proprioceptive input is the change in state (`obs.state[t] - obs.state[t-N]`) rather than the absolute state

These modifications follow the approach described in the UMI paper for handling relative representations.

## Key Differences from Standard ACT

### Input Representation
- **Standard ACT**: Uses absolute observation state `obs.state[t]`
- **ACT Relative RTC**: Uses observation delta `obs.state[t] - obs.state[t-N]` where N is configurable via `obs_state_delta_frames`

### Output Representation
- **Standard ACT**: Predicts absolute joint positions
- **ACT Relative RTC**: Predicts relative actions that are added to `obs.state[t]` to get absolute positions

### Normalization
- **Standard ACT**: Normalizes absolute observations and actions using dataset statistics
- **ACT Relative RTC**: Disables standard normalization for STATE and ACTION (set to IDENTITY), instead computing and applying normalization on the relative values (delta_obs, relative_action) internally

## Real-Time Chunking (RTC)

When `use_rtc=True`, the policy implements "training-time RTC" to improve chunk boundary consistency:

### Training (use_rtc=True)
- Per-sample delay is sampled uniformly from {0, ..., rtc_max_delay}
- Action prefix (first `delay` actions) is extracted from the chunk
- Prefix is padded to `rtc_max_delay` using learnable `pad_embed` parameter
- Prefix tokens are concatenated to encoder input (after latent, state, images)
- Prefix tokens participate in encoder self-attention (encoder_input strategy)
- Loss is computed only on postfix (positions >= delay per sample)
  - Note: With delay=0, loss is computed on all positions (no masking)

### Inference (use_rtc=True)
- `select_action()` is NOT supported; use `predict_action_chunk()` instead
- `predict_action_chunk()` accepts `delay` and `action_prefix` parameters
- `action_prefix` contains absolute actions from previous chunk
- Method converts to relative, normalizes, and passes to model
- Model uses prefix to condition prediction for smooth chunk transitions

### How It Works
1. During training, model learns to handle variable-length prefixes via per-sample delays
2. Learnable `pad_embed` (parameter) allows model to distinguish real prefix from padding
3. At inference, fixed delay from previous chunk provides consistent conditioning
4. Encoder processes prefix through self-attention, learning to incorporate prefix information

## Usage

### Training with lerobot-train (RTC Enabled)

```bash
lerobot-train \
    --policy.type act_relative_rtc \
    --dataset.repo_id your/dataset \
    --policy.chunk_size 50 \
    --policy.use_rtc true \
    --policy.rtc_max_delay 3 \
    --steps 50000
```

### Training without RTC (Default)

```bash
lerobot-train \
    --policy.type act_relative_rtc \
    --dataset.repo_id your/dataset \
    --policy.chunk_size 50 \
    --steps 50000
```

### Configuration Options

Key configuration parameters in `ACTRelativeRTCConfig`:

- `obs_state_delta_frames`: Number of frames back for computing observation delta (default: 1)
- `use_rtc`: Enable Real-Time Chunking (training-time action prefix conditioning). When enabled, the model is trained with random action prefixes to simulate inference delays, improving chunk boundary consistency (default: False)
 - `rtc_max_delay`: Maximum delay for RTC training. During training, per-sample delay is sampled uniformly from {0, ..., rtc_max_delay} (default: 3)
- `chunk_size`: Action chunk size (default: 100)
- `n_action_steps`: Number of actions to execute per inference (default: 100)

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │          ACTRelativeRTCPolicy       │
                    └─────────────────────────────────────┘
                                       │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  Normalizer   │           │  ACTRelativeRTC │           │  Normalizer     │
│  (delta_obs)  │           │     (model)     │           │ (rel_action)    │
└───────────────┘           └─────────────────┘           └─────────────────┘
        │                             │                             │
        └──────── register_buffer ────┴──────── register_buffer ────┘
                  (mean, std)                   (mean, std)

           RTC-only modules (when use_rtc=True):
           ┌─────────────────────────────────────────┐
           │  action_prefix_proj (Linear)        │  Projects action prefix to model dim
           │  pad_embed (Parameter)             │  Learnable padding token
           └─────────────────────────────────────────┘
```

### Relative Stats Computation

On first training (when `dataset_meta` is provided and stats are not already configured):
1. The policy recreates the dataset using `dataset_meta.repo_id` and `dataset_meta.root`
2. Iterates through the dataset to compute statistics on:
   - `delta_obs = obs[t] - obs[t-N]`
   - `relative_action = action - obs[t]`
3. Configures the `Normalizer` modules with computed mean/std
4. Stats are stored in registered buffers and saved with checkpoints

## Known Limitations

### 1. Redundant Image Loading

**Issue**: LeRobot's `resolve_delta_timestamps()` applies `observation_delta_indices` uniformly to ALL observation features. Since we need `[-N, 0]` for state delta computation, images also load 2 frames when only 1 is needed.

**Impact**: ~2x memory usage for image loading during training (wasteful but functional).

**Workaround**: The model uses only the last image frame (`batch[key][:, -1]`).

**Future Improvement**: Propose per-feature delta indices to LeRobot.

### 2. Dataset Recreation in __init__

**Issue**: When training from scratch, the policy must recreate the dataset in `__init__` to compute relative stats. This is redundant with LeRobot's dataset creation.

**Impact**: One-time cost during initialization. Subsequent training resumes and inference load stats from checkpoint automatically.

**Root Cause**: LeRobot's `make_policy()` only passes `dataset_meta`, not the full dataset.

## File Structure

```
lerobot_policy_act_relative_rtc/
├── __init__.py                          # Package exports
├── configuration_act_relative_rtc.py    # Policy configuration
├── modeling_act_relative_rtc.py         # Policy implementation
├── processor_act_relative_rtc.py        # Pre/post-processing pipelines
├── relative_stats.py                    # Relative statistics computation
└── README.md                            # This file
```

## References

- [ACT Paper](https://huggingface.co/papers/2304.13705): Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- [UMI Paper](https://umi-gripper.github.io/): Universal Manipulation Interface
- [LeRobot](https://github.com/huggingface/lerobot): Hugging Face robotics library

