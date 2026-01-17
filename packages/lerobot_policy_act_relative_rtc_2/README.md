# ACT Relative RTC Policy v2

A modified ACT (Action Chunking Transformer) policy that uses relative joint representations with full action prefix conditioning for improved generalization in robot manipulation tasks.

> **Note on Backward Compatibility**: This is a new package (`lerobot_policy_act_relative_rtc_2`) created to preserve backward compatibility with existing trained models in `lerobot_policy_act_relative_rtc`. The v2 implementation has breaking changes in the model architecture and cannot load v1 checkpoints.

## Overview

This is **version 2** of the ACT Relative RTC policy, with significant architectural improvements over v1:

### Key Features

1. **Relative Actions**: Actions are predicted relative to the current observation state (`action - obs.state[t]`) rather than as absolute joint positions
2. **Full Action Prefix Conditioning**: Uses the entire action prefix (all delay actions) as encoder input instead of proprioception deltas, providing richer trajectory information
3. **Single Observation Loading**: Loads only `obs[t]` instead of `[obs[t-N], obs[t]]`, improving memory efficiency
4. **Extended Action Sequence**: Skips `action[0]` (always ~0 for relative actions) and loads `action[1:max_delay+chunk_size+1]` for more meaningful action history

### V2 Improvements

- **Better Trajectory Signal**: Full action prefix provides velocity and trajectory information through self-attention
- **Memory Efficient**: ~15-20% reduction in GPU memory (single observation, no temporal dimension on images)
- **Cleaner Architecture**: No redundant image loading, simpler encoder input assembly
- **Flexible Conditioning**: Model learns to use as much or as little of the prefix as needed through self-attention

These modifications are inspired by the UMI paper's approach to relative representations, extended with full action prefix conditioning.

## Key Differences from Standard ACT

### Input Representation
- **Standard ACT**: Uses absolute observation state `obs.state[t]` in encoder
- **ACT Relative RTC v2**:
  - Loads only `obs[t]` (single observation)
  - Encoder input: `[latent, env_state, image_features, action_prefix_tokens]`
  - No robot state/proprioception in encoder (used only to compute relative actions)

### Output Representation
- **Standard ACT**: Predicts absolute joint positions
- **ACT Relative RTC v2**: Predicts relative actions that are added to `obs.state[t]` to get absolute positions

### Action Prefix Conditioning
- **Standard ACT**: No action prefix conditioning
- **ACT Relative RTC v2**: Full action prefix (all delay actions) added to encoder input when use_rtc=True, providing trajectory context through self-attention

### Data Loading
- **Standard ACT**: Loads actions `[0:chunk_size]`
- **ACT Relative RTC v2**: Loads actions `[1:rtc_max_delay+chunk_size+1]` (skips action[0] which is always ~0 for relative actions)

### Normalization
- **Standard ACT**: Normalizes absolute observations and actions using dataset statistics
- **ACT Relative RTC v2**: Disables standard normalization for STATE and ACTION (set to IDENTITY), instead computing and applying normalization on relative actions only (no delta_obs in v2)

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

- `use_rtc`: Enable Real-Time Chunking (training-time action prefix conditioning). When enabled, the model is trained with random action prefixes to simulate inference delays, improving chunk boundary consistency (default: False)
- `rtc_max_delay`: Maximum delay for RTC training. During training, per-sample delay is sampled uniformly from {0, ..., rtc_max_delay}. This also determines the action prefix length loaded for conditioning (default: 3)
- `chunk_size`: Action chunk size (default: 100)
- `n_action_steps`: Number of actions to execute per inference (default: 100)

Note: `obs_state_delta_frames` is kept for backward compatibility but is no longer used in v2 (state_delta_indices always returns [0])

## Architecture (V2)

```
                    ┌─────────────────────────────────────┐
                    │       ACTRelativeRTCPolicy v2       │
                    └─────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
           ┌─────────────────┐                  ┌─────────────────┐
           │  ACTRelativeRTC │                  │  Normalizer     │
           │     (model)     │                  │ (rel_action)    │
           └─────────────────┘                  └─────────────────┘
                                                         │
                                        register_buffer ─┘
                                             (mean, std)

           RTC-only modules (when use_rtc=True):
           ┌─────────────────────────────────────────┐
           │  action_prefix_proj (Linear)        │  Projects action prefix to model dim
           │  pad_embed (Parameter)             │  Learnable padding token
           └─────────────────────────────────────────┘

           Encoder Input (V2):
           ┌─────────────────────────────────────────┐
           │  [latent, env_state, images, prefix]  │
           │  (no robot_state/delta_obs)            │
           └─────────────────────────────────────────┘
```

### Relative Stats Computation (V2)

On first training (when `dataset_meta` is provided and stats are not already configured):
1. The policy recreates the dataset using `dataset_meta.repo_id` and `dataset_meta.root`
2. Iterates through the dataset to compute statistics on:
   - `relative_action = action - obs[t]` (V2: delta_obs removed)
3. Configures the `Normalizer` module with computed mean/std
4. Stats are stored in registered buffers and saved with checkpoints

## V2 Design Notes

### Why Skip action[0]?

For relative actions (`action - obs[t]`), `action[0]` represents the first action in the episode relative to the initial observation. In most robotic tasks:

- The robot starts at a known pose
- `action[0] ≈ initial_obs` in absolute coordinates
- Therefore, `action[0] - obs[0] ≈ 0` in relative coordinates

By skipping `action[0]` and loading from index 1 onwards, we:
1. Avoid wasting sequence length on a near-zero action
2. Get more meaningful action history for prefix conditioning
3. Align better with the actual task structure (actions cause state transitions)

### Action Prefix vs. Observation Delta

**V1 approach** (observation delta):
- Used `delta_obs = obs[t] - obs[t-N]` as encoder input
- Provides velocity information but only at a single timestep
- Requires loading 2 observations and computing deltas

**V2 approach** (action prefix):
- Uses full action prefix `actions[0:delay]` as encoder input
- Provides entire trajectory history through self-attention
- Model learns to weight different parts of the prefix adaptively
- More expressive than single delta observation

## Known Limitations

### 1. Dataset Recreation in __init__

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

