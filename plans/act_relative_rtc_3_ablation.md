# ACT Relative RTC 3 — Ablation Plan

Goal: evaluate a new policy variant (lerobot_policy_act_relative_rtc_3) that rebases RTC targets/prefixes to the **last prefix action** instead of the image-time observation, and compare against the current act_relative_rtc_2 behavior.

## 0) Scope + hypothesis
- Hypothesis: rebasing outputs to the “now” frame (end of prefix) removes delay-dependent frame shifts, improving stability and chunk stitching.
- Risk: the image is still from time t, so the model must use the prefix to bridge t→t+d. This could degrade accuracy for larger delays or imperfect tracking.

## 1) Variants to compare
A) **Baseline (RTC2)**
- Policy: `lerobot_policy_act_relative_rtc_2`
- Targets/prefix: relative to `obs_state_t` (image-time)

B) **RTC3-rebased**
- Policy: `lerobot_policy_act_relative_rtc_3`
- Targets: relative to last prefix action (state at `t+d`)
- Prefix: rebased to same reference so last prefix step becomes zero (or close), i.e., prefix deltas end at 0

C) **RTC3-rebased + offset token (optional but recommended)**
- Same as B, plus an explicit “frame offset” token that encodes the sum of prefix deltas (the shift from image-time to current frame). This preserves image-time alignment without forcing the model to infer it.

Keep A and B mandatory; C is a low-cost add-on if time allows.

## 2) Implementation plan (creates lerobot_policy_act_relative_rtc_3)
1. Copy policy package:
   - Duplicate `packages/lerobot_policy_act_relative_rtc_2` → `packages/lerobot_policy_act_relative_rtc_3`
   - Rename policy class + config: `ACTRelativeRTCPolicy` → `ACTRelativeRTC3Policy`, `ACTRelativeRTCConfig` → `ACTRelativeRTC3Config`
   - Update `name = "act_relative_rtc_3"`
   - Update package metadata (pyproject, __init__ exports) so `make_policy` can load it

2. **Training logic changes** in `modeling_act_relative_rtc.py`:
   - Current: `relative_actions = absolute_actions - obs_state_t`
   - New: for each delay d, define a base `base_d = action[:, d-1]` (or `obs_state_t` if d=0). Then
     - Targets: `relative_actions_d = absolute_actions[:, d:d+chunk_size] - base_d`
     - Prefix (for encoder tokens): `relative_prefix_d = absolute_actions[:, :max_delay] - base_d`
   - You’ll need to build per-delay targets (B, D, chunk, dim) explicitly rather than slicing a single relative sequence.
   - Normalize using a new stats key, e.g. `relative_action_rebased`.

3. **Inference logic changes** in `predict_action_chunk`:
   - When `delay > 0`, set base to last prefix action (absolute), not `obs_state_t`.
   - Convert prefix to relative wrt that base.
   - Convert predicted relative actions back to absolute using that same base.

4. **Eval script** (`eval_async_rtc.py`):
   - No logic change if policy handles base internally, but add a comment for RTC3 about base choice.
   - Ensure it still passes **absolute** action_prefix.

5. **Dataset stats**:
   - Compute new relative stats for RTC3 (rebased). Either:
     - Add `relative_action_rebased` to `preprocess_dataset.py` output, or
     - Recompute on the fly in RTC3 with a new key name.

## 3) Training protocol
- Dataset: same hand-guided dataset used for RTC2.
- Training seeds: 3 seeds per variant (A, B). If C is added, 1 seed to validate concept.
- Keep all hyperparams identical: chunk size, RTC max delay, backbone, optimizer, etc.
- Ensure the same train/val split.

## 4) Evaluation protocol
Quantitative:
- Offline validation loss (L1) as logged by training.
- If you have it: rollouts success rate and average task completion time.

Qualitative:
- Record 3–5 rollouts per variant for the same tasks.
- Watch for chunk boundary discontinuities, overshoot, or oscillations.

RTC stress test:
- Run evaluation at 2 different delays (e.g., 0 and max_delay) and compare smoothness/accuracy.

## 5) Decision criteria
- Promote RTC3 if it:
  - Matches or improves L1/val loss across seeds, and
  - Shows visibly smoother chunk transitions or higher task success at non-zero delays.
- If RTC3 degrades at higher delays, try variant C (offset token) before discarding.

## 6) Concrete next steps
1. Implement `lerobot_policy_act_relative_rtc_3` package.
2. Add rebased stats computation and ensure checkpoints store them.
3. Train A and B (3 seeds) with identical configs.
4. Evaluate both with `eval_async_rtc.py` using RTC3 policy for B.
5. Compare metrics + videos and decide.
