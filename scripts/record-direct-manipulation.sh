#!/usr/bin/env bash
#
# Record episodes using direct manipulation (UMI-style, leader arm only).
#
# Usage:
#   ./scripts/record-direct-manipulation.sh --task=cube          # Start fresh recording
#   ./scripts/record-direct-manipulation.sh --task=cube --resume # Resume recording on existing dataset
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - leader.port: USB port for your leader arm (find with `lerobot-find-port`)
#   - leader.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

RESUME_FLAG=""
TASK=""
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME_FLAG="--resume=true" ;;
        --task=*) TASK="${arg#--task=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

case "$TASK" in
    cube) TASK_DESC="Pick up the cube and place it in the target location" ;;
    gba)  TASK_DESC="Press the up arrow on the GBA" ;;
    ball) TASK_DESC="Throw the ping-pong ball into the basket" ;;
    *)    echo "Usage: $0 --task={cube,gba,ball} [--resume]"; exit 1 ;;
esac

CAMERAS="{
  wrist: { type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, rotation: -90 }
}"

python -m so101_direct_manipulation.record.record \
    $RESUME_FLAG \
    --leader.id=leader \
    --leader.port=/dev/tty.usbmodem575E0080981 \
    --leader.cameras="$CAMERAS" \
    --dataset.repo_id=giacomoran/"so101_${TASK}"_direct_manipulation \
    --dataset.single_task="$TASK_DESC" \
    --dataset.num_episodes=100 \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --display_data=false
