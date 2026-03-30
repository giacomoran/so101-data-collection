#!/usr/bin/env bash
#
# Record episodes using standard teleoperation (leader + follower arms).
#
# Usage:
#   ./scripts/record-teleoperation.sh --task=cube          # Start fresh recording
#   ./scripts/record-teleoperation.sh --task=cube --resume # Resume recording on existing dataset
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port / teleop.port: USB ports for your arms (find with `lerobot-find-port`)
#   - robot.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

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
  wrist: { type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90 },
  top:   { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }
}"

lerobot-record \
    $RESUME_FLAG \
    --robot.type=so101_follower \
    --robot.id=arm_follower_0 \
    --robot.port=/dev/tty.usbmodem5A460829821 \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.id=arm_leader_0 \
    --teleop.port=/dev/tty.usbmodem5A460824651 \
    --dataset.repo_id=giacomoran/so101_"${TASK}"_teleoperation \
    --dataset.single_task="$TASK_DESC" \
    --dataset.num_episodes=50 \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --display_data=false
