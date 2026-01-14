# GPU Utilization Debugging Plan

## Goal

Fix the poor GPU utilization in `act_relative_rtc` training. Standard `act` policy achieves good GPU utilization, but `act_relative_rtc` shows terrible utilization on the same hardware.

**Baseline Results** (confirmed):
- Standard ACT: Good GPU utilization (>80%)
- act_relative_rtc: Poor GPU utilization (<30%)

## Problem Context

The `act_relative_rtc` policy extends standard ACT with:
1. **Relative transformations**: Converts observations to deltas and actions to relative values
2. **Double image loading**: Due to `delta_timestamps` applying uniformly to all observation features, images load as `[B, 2, C, H, W]` but only `[:, -1]` is used (2x I/O overhead)
3. **Additional preprocessing**: Image resize/pad operations and relative stats normalization

The preprocessed dataset `so101_data_collection_cube_hand_guided_1x224x8` already has images resized to 224x224, so runtime image preprocessing is not the issue.

## Your Task

As a coding agent, you will:
1. Read the progress file at `outputs/tmp/gpu_debug_progress.md`
2. If baselines are missing, record them from the completed runs
3. If baselines exist, come up with an hypothesis which has not been tested already
4. Implement and test the hypothesis
5. Update the progress file with results
6. Stop (next agent will continue)

## How to Run Checks

### Test a Hypothesis

Run the act_relative_rtc training with your modifications using the GPU monitoring script:

```bash
python packages/so101_data_collection/src/so101_data_collection/train/debug_gpu_utilization.py --output outputs/tmp/gpu_test_hypothesis_N.csv --cmd 'lerobot-train \
    --policy.type=act_relative_rtc \
    --dataset.repo_id=giacomoran/so101_data_collection_cube_hand_guided_1x224x8 \
    --dataset.episodes=[0] \
    --steps=200 \
    --batch_size=8 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=8 \
    --policy.n_action_steps=8 \
    --policy.obs_state_delta_frames=1 \
    --policy.use_vae=false \
    --policy.n_decoder_layers=4 \
    --policy.downscale_img_square=224 \
    --policy.vision_backbone=resnet34 \
    --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 \
    --policy.pre_norm=true \
    --policy.use_rtc=false \
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
- **Baseline act_relative_rtc**: Check progress file for the baseline GPU util
- **Your test**: GPU util from your run
- **Improvement**: Calculate the difference

## Success Criteria

A hypothesis **succeeds** when:
- GPU utilization improves by **≥20 percentage points** vs baseline act_relative_rtc
- Training completes without errors
- Loss decreases over training steps (not NaN/diverging)

A hypothesis **fails** when:
- GPU utilization improves by <10 percentage points
- Training crashes or produces errors
- Loss is NaN or diverges

Note: Loss values are not comparable between ACT and act_relative_rtc due to different architectures.

## Progress File Format

Update `outputs/tmp/gpu_debug_progress.md` after testing:

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
- `packages/lerobot_policy_act_relative_rtc/src/lerobot_policy_act_relative_rtc/`
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

1. Update `outputs/tmp/gpu_debug_progress.md` with your results
2. If hypothesis succeeded: Document the fix and suggest next steps
3. If hypothesis failed: Document why and move to next hypothesis
4. **Stop** - Do not test multiple hypotheses in one session

The next coding agent will read the progress file and continue with the next hypothesis.
