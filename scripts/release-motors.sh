#!/usr/bin/env bash

python -m so101_direct_manipulation.shared.release_motors \
    --port=/dev/tty.usbmodem5A460829821 --id=follower \
    --port=/dev/tty.usbmodem5A460824651 --id=leader_direct_manipulation
