#!/usr/bin/env bash
#
# Create a calibration profile for the follower arm.
#
# Usage:
#   ./scripts/calibrate-follower.sh
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port: USB port for your follower arm (find with `lerobot-find-port`)

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.id=follower \
    --robot.port=/dev/tty.usbmodem5A460829821
