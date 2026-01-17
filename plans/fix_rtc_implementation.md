# Fix RTC Implementation Plan

## Overview

Transform ACT Relative RTC to:

- **Load single observation** instead of two (might fix GPU utilization)
- **Skip action[0]** (always ~0 for relative actions) and load extended sequence
- **Use full action prefix** instead of proprioception delta
- **Pass entire action prefix to encoder** for richer trajectory conditioning

## Key Design

### Data Loading Changes

- **Observations**: Load only `obs[t]` with shape `[batch, state_dim]` (not `[obs[t-N], obs[t]]`)
- **Actions**: Load `[action[1:max_rtc_delay+chunk_size+1]]` with shape `[batch, rtc_max_delay+chunk_size, action_dim]`
  - Prefix: `actions[0:delay]` (indices 1 to delay in episode)
  - Targets: `actions[delay:delay+chunk_size]` (indices delay+1 to delay+chunk_size in episode)

### Encoder Input

- **OLD**: `[latent, delta_obs, env_state, image_features, action_prefix_tokens]`
- **NEW**: `[latent, env_state, image_features, action_prefix_tokens]`
- Proprioception (`obs.state`) is NOT fed to encoder, only used to compute relative actions
- Full action prefix (all delay actions) provides velocity and trajectory information

### Edge Cases

- **delay=0**: No action prefix tokens added to encoder (no prefix available)
- **First chunk**: Pass `delay=0, action_prefix=None` to model

## Critical Files to Modify

### 1. configuration_act_relative_rtc.py (~30 lines)

**Changes:**

- `state_delta_indices`: Return `[0]` instead of `[-obs_state_delta_frames, 0]`
- `action_delta_indices`: Return `list(range(1, rtc_max_delay + chunk_size + 1))`
- Add validation: `rtc_max_delay >= 1` when `use_rtc=True`

**Location**: Lines 195-235

### 2. modeling_act_relative_rtc.py (~300 lines)

This is the most complex file with major refactoring needed.

#### 2.1 Remove delta_obs Normalizer (Lines 147-150)

```python
# REMOVE: self.delta_obs_normalizer = Normalizer(state_dim)
# KEEP: self.relative_action_normalizer = Normalizer(action_dim)
```

#### 2.2 Update Properties and Methods

- `has_relative_stats` (Line 299): Check only `relative_action_normalizer.is_configured`
- `set_relative_stats()` (Line 304): Remove delta_obs configuration
- `reset()` (Line 326): Remove observation queue (no longer needed)

#### 2.3 Rewrite forward() Method (Lines 457-568)

**New flow:**

1. Extract `obs_state_t = batch[OBS_STATE]` (now `[batch, state_dim]`)
2. Extract `absolute_actions = batch[ACTION]` (now `[batch, rtc_max_delay+chunk_size, action_dim]`)
3. Compute `relative_actions_extended = absolute_actions - obs_state_t.unsqueeze(1)`
4. Normalize with `relative_action_normalizer`
5. Split into prefix and targets:
   - `action_prefix = relative_actions_extended[:, :max_delay]`
   - `targets = relative_actions_extended[:, max_delay:max_delay+chunk_size]`
6. Sample delays, mask prefix, pass to model
7. Compute loss on targets with proper masking

**Key**: Remove all delta_obs computation, handle extended action sequence properly

#### 2.4 Rewrite ACTRelativeRTC.forward() (Lines 695-816)

**New signature:**

```python
def forward(
    self,
    batch: dict[str, Tensor],
    action_prefix: Tensor | None = None,
    delays: Tensor | None = None,
    obs_state: Tensor | None = None,  # NEW: for future use
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
```

**New encoder input assembly:**

1. Build tokens: `[latent, env_state, image_features]` (NO robot_state/delta_obs)
2. Add full action prefix based on delay:

   ```python
   if use_rtc and action_prefix is not None and delays is not None:
       # Project action prefix embeddings
       action_prefix_embed = self.action_prefix_proj(action_prefix)  # [batch, rtc_max_delay, dim_model]

       # Mask padding positions with pad_embed
       # For sample i, positions >= delays[i] should use pad_embed
       mask = torch.arange(rtc_max_delay, device=delays.device)[None, :] >= delays[:, None]
       action_prefix_embed = torch.where(
           mask.unsqueeze(-1),
           self.pad_embed[None, None, :].expand_as(action_prefix_embed),
           action_prefix_embed,
       )

       # Add all prefix tokens to encoder input
       encoder_in_tokens.extend(list(action_prefix_embed.permute(1, 0, 2)))
   ```

3. Add sinusoidal positional embeddings for prefix tokens

**VAE Encoder Changes (Lines 720-747):**

- Remove robot_state from VAE encoder input
- VAE input: `[cls_embed, action_embed]` only

**Main Encoder Init Changes (Lines 661-673):**

- Remove `self.encoder_robot_state_input_proj`
- Update `n_1d_tokens` calculation (no robot_state token)

**VAE Encoder Init Changes (Lines 626-644):**

- Remove `self.vae_encoder_robot_state_input_proj`
- Update `num_input_token_encoder = 1 + chunk_size` (no robot_state)

#### 2.5 Update Inference Methods

**select_action() (Lines 341-408):**

- Remove observation queue logic
- Get `obs_state_t` directly from batch (single observation)
- Pass `obs_state=obs_state_t` to `predict_action_chunk()`
- Convert predicted relative actions to absolute

**predict_action_chunk() (Lines 410-455):**

- Add `obs_state` parameter (required)
- Convert action_prefix from absolute to relative internally
- Remove delta_obs handling

### 3. relative_stats.py (~50 lines)

**Changes to compute_relative_stats() (Lines 27-128):**

**Remove:**

- All delta_obs computation
- `delta_obs_stats` from return value

**Update:**

1. Load `obs_state_t = batch[OBS_STATE]` (now `[batch, state_dim]`)
2. Load `absolute_actions = batch[ACTION]` (now `[batch, rtc_max_delay+chunk_size, action_dim]`)
3. Compute `relative_actions = absolute_actions - obs_state_t.unsqueeze(1)`
4. Handle `action_is_pad` for extended sequence
5. Return only: `{"relative_action": {"mean": ..., "std": ...}}`

### 4. processor_act_relative_rtc.py (~20 lines)

**Update DropUnusedImageFramesProcessorStep (Lines 53-96):**

**Convert to no-op:**

```python
def observation(self, observation: dict) -> dict:
    # No-op: images already have correct shape [B, C, H, W]
    # (state_delta_indices = [0] means no temporal dimension)
    return observation
```

**Update docstring:** Explain this is now a no-op due to conditioning on the action prefix

### 5. Documentation Updates

**README.md:**

- Update overview to explain new conditioning
- Remove "Redundant Image Loading" limitation (fixed!)
- Add section on why action[0] is skipped
- Update architecture diagram

## Implementation Sequence

1. **Configuration** (configuration_act_relative_rtc.py)
   - Update delta indices properties
   - Add validation

2. **Statistics** (relative_stats.py)
   - Remove delta_obs computation
   - Test on small dataset

3. **Model Init** (modeling_act_relative_rtc.py)
   - Remove delta_obs_normalizer
   - Update VAE/encoder initialization
   - Update properties/methods

4. **Forward Pass** (modeling_act_relative_rtc.py)
   - Rewrite ACTRelativeRTCPolicy.forward()
   - Rewrite ACTRelativeRTC.forward()
   - Test training loop

5. **Inference** (modeling_act_relative_rtc.py)
   - Update select_action()
   - Update predict_action_chunk()
   - Test inference

6. **Processor** (processor_act_relative_rtc.py)
   - Convert DropUnusedImageFramesProcessorStep to no-op

7. **Documentation** (README.md)
   - Update all relevant sections

## Testing Strategy

### Unit Tests

- **Data shapes**: Verify obs=[batch, state_dim], actions=[batch, rtc_max_delay+chunk_size, action_dim]
- **Forward pass**: Test all delay values from 0 to rtc_max_delay
- **Prefix embedding**: Verify all prefix actions are correctly embedded
- **Edge cases**: delay=0, padded sequences

### Integration Tests

- Train for 100 steps, verify loss decreases
- Run inference with various delays
- Compare memory usage (expect ~15-20% reduction)

### Validation

- Compare training curves with baseline
- Verify task performance
- Benchmark GPU utilization

## Edge Cases to Handle

### delay=0 (No Prefix)

- Training: No action prefix tokens added to encoder
- Loss computed on all chunk_size positions
- Model learns to predict without prefix conditioning

### First Chunk (Inference)

- Pass `delay=0, action_prefix=None`
- No action prefix tokens added to encoder

### Action Padding Mask

- Extract from extended sequence:
  ```python
  action_is_pad_extended = batch["action_is_pad"]  # [batch, rtc_max_delay+chunk_size]
  action_is_pad = action_is_pad_extended[:, max_delay:max_delay+chunk_size]
  ```

### Images Without Temporal Dimension

- With `state_delta_indices=[0]`, images load as `[B, C, H, W]`
- No squeezing needed in forward()
- DropUnusedImageFramesProcessorStep becomes no-op

## Checkpoint Compatibility

**BREAKING CHANGES** - old checkpoints cannot be loaded:

**Removed parameters:**

- `delta_obs_normalizer.mean`
- `delta_obs_normalizer.std`
- `delta_obs_normalizer._is_configured`
- `vae_encoder_robot_state_input_proj.weight`
- `vae_encoder_robot_state_input_proj.bias`
- `encoder_robot_state_input_proj.weight`
- `encoder_robot_state_input_proj.bias`

**Migration:** Retrain from scratch

## Expected Benefits

1. **Memory Efficiency**: ~15-20% reduction in GPU memory
   - Single observation instead of two
   - No temporal dimension on images

2. **Cleaner Architecture**:
   - No redundant image loading
   - Simpler encoder input assembly

3. **Better Trajectory Signal**:
   - Full action prefix provides richer velocity and trajectory information
   - More context than single delta_obs observation

4. **Flexible Conditioning**:
   - Model can learn to use as much or as little of the prefix as needed
   - Self-attention over prefix allows adaptive weighting

## Design Notes

The architecture uses:

- `action_prefix_proj`: Linear projection for action embeddings (all prefix actions)
- `pad_embed`: Learnable padding token for positions >= delay
- Full prefix conditioning: All actions in prefix [0:delay-1] are added to encoder input
- Self-attention: Model can learn to weight different parts of the prefix differently

This is more expressive than using only the last action, as the model can learn temporal patterns across the entire prefix through self-attention.

## Verification Checklist

After implementation:

- [ ] Data loads with correct shapes (single obs, extended actions)
- [ ] Training runs without errors for all delay values
- [ ] Loss decreases over training
- [ ] Inference works with delay=0 and delay>0
- [ ] Memory usage reduced by ~15-20%
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation updated
