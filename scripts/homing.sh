#!/usr/bin/env bash

python -m so101_direct_manipulation.shared.homing \
    --port=/dev/tty.usbmodem5A460829821 \
    --id=arm_follower_0
