# Plan: Create eval_async_rtc.py

## Goal
Create `eval_async_rtc.py` that runs async inference with Real-Time Chunking (RTC) support for ACTRelativeRTC policies trained with `use_rtc=True`.

## Background

### How RTC Works
- During training, the policy is conditioned on an **action prefix** (the first `delay` steps of the action chunk)
- The policy learns to generate a postfix that is **consistent** with the prefix
- At inference time, we pass the actions being executed during inference as the prefix
- Result: smooth chunk transitions without discontinuities

### Key Interface (from ACTRelativeRTC)
```python
policy.predict_action_chunk(
    batch,                    # includes normalized delta observation
    delay=delay,              # number of prefix steps
    action_prefix=prefix      # shape [batch, delay, action_dim], MUST be normalized relative actions
)
```

The caller is responsible for:
1. Converting absolute actions to relative: `relative = absolute - obs_state_t`
2. Normalizing the relative prefix: `normalized = policy.relative_action_normalizer(relative)`

## Implementation Approach

### Architecture (same as eval_async_discard.py)
- **Main thread**: monitors for termination
- **Actor thread**: executes actions at target fps, triggers inference
- **Inference thread**: runs policy inference when requested

### Key Differences from eval_async_discard.py

1. **When triggering inference**, pass the action prefix:
   - The prefix = next `delay` actions from current chunk that will execute during inference
   - Convert to relative actions using the observation state at trigger time
   - Normalize the relative prefix

2. **New data in shared state**:
   - `inference_obs_state_t`: observation state at inference trigger time (for relative conversion)
   - `inference_action_prefix`: absolute actions to use as prefix
   - `inference_delay`: number of prefix steps

3. **In inference thread**:
   - Convert absolute prefix to normalized relative prefix
   - Call `predict_action_chunk(batch, delay=delay, action_prefix=normalized_prefix)`
   - The returned chunk is already consistent with the prefix

4. **Delay computation**:
   - Use config `rtc_delay` (default: `rtc_max_delay` from policy config)
   - Or estimate from inference latency: `ceil(inference_ms / dt_ms)`

## Critical Files to Modify/Create

- **Create**: `packages/so101_data_collection/src/so101_data_collection/eval/eval_async_rtc.py`
- **Reference**: `packages/so101_data_collection/src/so101_data_collection/eval/eval_async_discard.py`
- **Reference**: `packages/lerobot_policy_act_relative_rtc/src/lerobot_policy_act_relative_rtc/modeling_act_relative_rtc.py`

## Detailed Implementation

### 1. Config (EvalAsyncRTCConfig)
```python
@dataclass
class EvalAsyncRTCConfig:
    # ... same fields as EvalAsyncDiscardConfig ...

    # RTC-specific: number of prefix steps to use for conditioning
    # If None, defaults to policy.config.rtc_max_delay
    # User can tune this for experimentation (e.g., match actual inference latency)
    rtc_delay: int | None = None
```

At runtime, resolve to: `rtc_delay if rtc_delay is not None else policy.config.rtc_max_delay`

### 2. Modified State dataclass
```python
@dataclass
class State:
    # ... existing fields from eval_async_discard.py ...

    # RTC: data for prefix conditioning
    # Set by actor when triggering inference
    inference_obs_state_t: torch.Tensor | None = None  # [1, state_dim] - obs at trigger time
    inference_action_prefix: torch.Tensor | None = None  # [1, delay, action_dim] - absolute actions
    inference_delay: int = 0
```

### 3. Actor thread modifications

When triggering inference:
```python
# Current code in eval_async_discard.py:
if not state.inference_running and remaining < cfg.remaining_actions_threshold:
    state.inference_running = True
    state.inference_requested.set()

# New code for RTC:
if not state.inference_running and remaining < cfg.remaining_actions_threshold:
    # Compute delay (how many actions will execute during inference)
    delay = min(rtc_delay, remaining)

    # Extract action prefix (next `delay` actions from current chunk)
    # These are the actions that will execute during inference
    action_prefix = current_chunk.actions[effective_t - current_chunk.obs_timestep:][:delay]
    action_prefix = action_prefix.unsqueeze(0)  # add batch dim

    # Store for inference thread
    state.inference_obs_state_t = obs_state.clone()  # observation state at trigger time
    state.inference_action_prefix = action_prefix
    state.inference_delay = delay

    state.inference_running = True
    state.inference_requested.set()
```

### 4. Inference thread modifications

```python
# Get RTC data
with state.lock:
    raw_obs = state.current_obs
    obs_timestep = state.obs_timestep
    delay = state.inference_delay
    action_prefix_absolute = state.inference_action_prefix  # [1, delay, action_dim]
    obs_state_t = state.inference_obs_state_t  # [1, state_dim]

    # ... compute delta_obs from history as before ...

# Convert absolute prefix to relative, then normalize
if action_prefix_absolute is not None and delay > 0:
    # relative = absolute - obs_state_t
    action_prefix_relative = action_prefix_absolute - obs_state_t.unsqueeze(1)

    # Normalize if stats available
    if policy.has_relative_stats:
        action_prefix_relative = policy.relative_action_normalizer(action_prefix_relative)
else:
    action_prefix_relative = None
    delay = 0

# Call policy with RTC params
relative_actions_normalized = policy.predict_action_chunk(
    inference_batch,
    delay=delay,
    action_prefix=action_prefix_relative,
)
```

### 5. Chunk transition handling

The actor thread chunk switching logic remains mostly the same. Since the new chunk was conditioned on the prefix:
- Positions 0..delay-1 in new chunk are consistent with what was executed
- The transition to position `delay` (first postfix action) should be smooth

## Edge Cases

1. **First chunk (no previous actions)**: When `current_chunk is None`, trigger inference with `delay=0` and `action_prefix=None`. This matches vanilla inference.

2. **Not enough remaining actions for prefix**: If `remaining < rtc_delay`, use `delay = remaining` and only pass available actions as prefix.

3. **Policy without RTC support**: Check `hasattr(policy.config, 'rtc_max_delay')`. If not present, fall back to `delay=0` (vanilla async).

## Verification

1. **Manual test**: Run with a trained RTC model:
   ```bash
   python -m so101_data_collection.eval.eval_async_rtc \
       --robot.type=so101_follower \
       --robot.port=/dev/tty.usbmodem... \
       --robot.cameras="{wrist: {...}}" \
       --policy.path=outputs/model_with_rtc/pretrained_model \
       --dataset_repo_id=giacomoran/cube_hand_guided \
       --fps=30 \
       --rtc_delay=3 \
       --display_data=true
   ```

2. **Compare with eval_async_discard.py**:
   - Run both on same policy (trained with use_rtc=True)
   - Observe chunk transition smoothness in rerun visualizations
   - RTC should show smoother action trajectories at chunk boundaries
