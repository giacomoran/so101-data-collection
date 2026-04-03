#!/usr/bin/env bash
#
# Evaluate ACTSmooth policy on cube direct manipulation task.
#
# Policy: ACTSmooth p4f2 @ 30fps (trained with train-cube-direct-manipulation.sh)
#
# Usage:
#   ./scripts/eval-cube-direct-manipulation.sh --policy.path=/path/to/pretrained_model
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port: USB port for your follower arm (find with `lerobot-find-port`)
#   - robot.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

python -m so101_direct_manipulation.eval.eval_async_smooth \
    --robot.type=so101_follower \
    --robot.id=follower \
    --robot.port=/dev/tty.usbmodem5A460829821 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90}}" \
    --fps_policy=30 \
    --fps_interpolation=30 \
    --fps_observation=60 \
    --duration_s_episode=60 \
    --is_logging_rr=false \
    "$@"
