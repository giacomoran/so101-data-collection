# RTC Batch Expansion: Evaluate All Delays in Single Forward Pass

## Context

Currently, `ACTRelativeRTCPolicy.forward()` samples one random delay per sample:

```python
delays = torch.randint(0, max_delay + 1, (batch_size,), device=...)
```

This means each training step only gets supervision from one delay value per sample. The VFLASH paper shows that evaluating all delays improves training efficiency.

## Goal

Modify `forward()` to evaluate all `D = rtc_max_delay + 1` delays per sample in a single forward pass, while:

1. Running the expensive image backbone only ONCE
2. Not copying image tensors in memory
3. Getting D× supervision signal per batch

## Benchmark Results

With `rtc_max_delay=3` (D=4), ResNet34, 224×224, batch_size=32:

- Backbone: 64.3% of forward pass (27ms)
- Encoder + Decoder + Other: 35.7% (15ms)

Expected overhead: ~2× original cost for 4× supervision signal.

## Implementation

### Memory Strategy

**Critical**: Use `expand()` (creates view, no copy) instead of `repeat()` (copies data).

```python
# GOOD: expand creates a view with stride=0, no memory allocation
x_expanded = x.unsqueeze(1).expand(B, D, -1, -1).reshape(B * D, ...)

# BAD: repeat allocates new memory
x_repeated = x.unsqueeze(1).repeat(1, D, 1, 1).reshape(B * D, ...)
```

After `reshape()`, PyTorch may need to make contiguous, but:

- Backbone output is small: [B, 49, 256] ≈ 12K floats per sample (vs 150K for input image)
- This is acceptable to copy if needed

### Changes to `ACTRelativeRTC.forward()`

Location: `modeling_act_relative_rtc.py`, class `ACTRelativeRTC`, method `forward()`

#### Step 1: Run backbone ONCE (no changes to this part)

```python
if self.config.image_features:
    for img in batch[OBS_IMAGES]:
        cam_features = self.backbone(img)["feature_map"]  # [B, C, H, W]
        # ... projection and position embedding
```

This stays the same - backbone runs on [B, ...] tensors.

#### Step 2: After backbone, before encoder - expand for all delays

Insert expansion logic after image features are processed but before encoder:

```python
# After collecting encoder_in_tokens and encoder_in_pos_embed...

if is_training_batch:
    D = max_delay + 1  # number of delays to evaluate

    # Stack tokens: [N_tokens, B, dim] → [N_tokens, B, 1, dim] → [N_tokens, B, D, dim]
    # Then reshape to [N_tokens, B*D, dim]
    encoder_in_tokens = encoder_in_tokens.unsqueeze(2).expand(-1, -1, D, -1)
    encoder_in_tokens = encoder_in_tokens.reshape(encoder_in_tokens.shape[0], B * D, -1)

    # Same for pos_embed
    encoder_in_pos_embed = encoder_in_pos_embed.unsqueeze(2).expand(-1, -1, D, -1)
    encoder_in_pos_embed = encoder_in_pos_embed.reshape(encoder_in_pos_embed.shape[0], B * D, -1)

    # Create delays tensor: [0,1,2,3,0,1,2,3,...] for each sample
    delays = torch.arange(D, device=device).unsqueeze(0).expand(B, D).reshape(B * D)

    # Expand actions: [B, seq_len, action_dim] → [B*D, seq_len, action_dim]
    actions = actions.unsqueeze(1).expand(B, D, -1, -1).reshape(B * D, -1, actions.shape[-1])
```

#### Step 3: Process action prefix with per-delay masking

The existing prefix masking logic should work unchanged since `delays` now contains the correct per-virtual-sample delay:

```python
# This already handles variable delays per sample
mask = torch.arange(max_delay, device=delays.device)[None, :] >= delays[:, None]
action_prefix_embed = torch.where(
    mask.unsqueeze(-1),
    self.pad_embed[None, None, :].expand_as(action_prefix_embed),
    action_prefix_embed,
)
```

#### Step 4: Run encoder and decoder (unchanged)

```python
encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
# ... decoder runs on [B*D, ...] tensors
```

#### Step 5: Reshape output and compute loss

Back in `ACTRelativeRTCPolicy.forward()`:

```python
pred_relative_actions, (mu_hat, log_sigma_x2_hat) = self.model(training_batch)
# pred_relative_actions: [B*D, chunk_size, action_dim]

if is_training_with_all_delays:
    # Reshape predictions: [B*D, chunk_size, action_dim] → [B, D, chunk_size, action_dim]
    pred_relative_actions = pred_relative_actions.reshape(B, D, chunk_size, -1)

    # Expand targets to match: [B, chunk_size, action_dim] → [B, D, chunk_size, action_dim]
    targets = targets.unsqueeze(1).expand(B, D, chunk_size, -1)

    # Expand padding mask
    action_is_pad = action_is_pad.unsqueeze(1).expand(B, D, chunk_size)

    # Compute loss over all delays (mean reduction handles it)
    pad_mask = ~action_is_pad.unsqueeze(-1)
    l1_loss_full = F.l1_loss(targets, pred_relative_actions, reduction="none")
    l1_loss = (l1_loss_full * pad_mask).mean()
```

### Config Changes

Add a flag to enable/disable this optimization (for A/B testing):

```python
# In ACTRelativeRTCConfig
rtc_train_all_delays: bool = True  # If True, evaluate all delays; if False, sample one
```

### VAE Encoder Handling

The VAE encoder (if `use_vae=True`) also needs to be expanded. It processes action targets, so:

```python
if self.config.use_vae and is_training_batch and self.training:
    if train_all_delays:
        # Expand action_targets for VAE: [B, chunk_size, dim] → [B*D, chunk_size, dim]
        action_targets = action_targets.unsqueeze(1).expand(B, D, -1, -1).reshape(B * D, -1, action_dim)
        action_is_pad_targets = action_is_pad_targets.unsqueeze(1).expand(B, D, -1).reshape(B * D, -1)
```

## Memory Analysis

With `batch_size=32`, `rtc_max_delay=3` (D=4):

- Effective batch for transformer: 32 × 4 = 128
- Image tensors: NOT expanded (backbone runs on B=32)
- Backbone output: ~12K floats/sample × 32 = 384K floats (1.5MB) - expanded to 6MB
- Actions: ~60 × 6 × 32 = 11.5K floats (46KB) - expanded to 184KB

Total additional memory: ~5MB for backbone features + negligible for actions. This is acceptable.

## Testing

1. **Correctness**: Verify that with `rtc_train_all_delays=False`, behavior matches current implementation exactly.

2. **Gradient check**: Ensure gradients flow correctly through the expanded computation.

3. **Performance**: Benchmark forward+backward pass time with flag on vs off.

4. **Training**: Compare loss curves with same number of gradient steps.

You can make sure that the model can be trained correctly by running:

```
lerobot-train \
  --policy.type=act_relative_rtc_2 \
  --dataset.repo_id=giacomoran/so101_data_collection_cube_hand_guided_1x224x8 \
  --dataset.episodes=[0] \
  --policy.repo_id=giacomoran/so101_data_collection_cube_hand_guided_act_wrist_20_overfit \
  --output_dir=outputs/cube_hand_guided_act_wrist_20_overfit \
  --policy.input_features='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
  --steps=500 \
  --save_freq=10000 \
  --batch_size=32 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_lr_backbone=3e-5 \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --policy.use_vae=false \
  --policy.n_decoder_layers=4 \
  --policy.downscale_img_square=224 \
  --policy.vision_backbone=resnet34 \
  --policy.pretrained_backbone_weights='ResNet34_Weights.IMAGENET1K_V1' \
  --policy.pre_norm=true \
  --policy.rtc_max_delay=3 \
  --policy.device=cuda \
  --wandb.enable=false \
  --wandb.disable_artifact=false \
  --policy.push_to_hub=false
```

## Rollback

If issues arise, set `rtc_train_all_delays=False` to revert to original behavior.

## Files to Modify

1. `configuration_act_relative_rtc.py`: Add `rtc_train_all_delays` config option
2. `modeling_act_relative_rtc.py`:
   - `ACTRelativeRTC.forward()`: Add expansion logic
   - `ACTRelativeRTCPolicy.forward()`: Handle reshaped outputs and loss computation

## Open Questions

1. Should we weight delays differently? (e.g., higher weight for delay=0 since that's inference mode)
2. Should we report per-delay loss in `loss_dict` for monitoring?
