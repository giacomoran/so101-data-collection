#!/usr/bin/env bash
#
# Create a calibration profile for the leader arm for direct manipulation.
#
# Usage:
#   ./scripts/calibrate-leader-direct-manipulation.sh
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - leader.port: USB port for your leader arm (find with `lerobot-find-port`)

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.id=leader_direct_manipulation \
    --teleop.port=/dev/tty.usbmodem5A460824651
