#!/usr/bin/env python3
"""
Release (disable torque on) all motors on the SO-101 robot arms.

Can be used as a module:
    from so101_direct_manipulation.shared.release_motors import release_motors

    release_motors(robot)  # robot is a connected SO101Follower

Or as a standalone script:
    python -m so101_direct_manipulation.shared.release_motors --port=/dev/tty.usbmodem...
"""

import argparse
import logging
import sys

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

logger = logging.getLogger(__name__)


def release_motors(robot: SO101Follower) -> None:
    """
    Disable torque on all motors of a connected SO101Follower.

    Args:
        robot: A connected SO101Follower instance.
    """
    robot.bus.disable_torque()
    logger.info("Torque disabled on all motors")


def main() -> int:
    """Release motors on one or more SO-101 arms."""
    parser = argparse.ArgumentParser(description="Release (disable torque on) SO-101 robot motors")
    parser.add_argument(
        "--port",
        type=str,
        action="append",
        required=True,
        help="Robot serial port(s) to release (can be specified multiple times)",
    )
    parser.add_argument("--id", type=str, default="arm_follower_0", help="Robot ID prefix")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    failed = 0
    for i, port in enumerate(args.port):
        robot_id = f"{args.id}_{i}" if len(args.port) > 1 else args.id
        logger.info(f"Releasing motors on {port} (id={robot_id})")

        robot_config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
        )
        robot = SO101Follower(robot_config)

        try:
            robot.connect()
            release_motors(robot)
            robot.disconnect()
            logger.info(f"Done: {port}")
        except Exception as e:
            logger.error(f"Failed on {port}: {e}")
            failed += 1

    if failed:
        logger.warning(f"{failed}/{len(args.port)} arms failed to release")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
