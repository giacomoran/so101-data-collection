#!/usr/bin/env python3
"""
Release (disable torque on) all motors on the SO-101 robot arms.

This script connects to the follower and leader arms and disables torque,
allowing the arms to be moved freely by hand.
"""

import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from so101_data_collection.shared.setup import LEADER_PORT, ROBOT_PORT

# Motor configuration for SO-101 arms (same for leader and follower)
SO101_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def release_motors(port: str, name: str) -> bool:
    """
    Connect to a motor bus and disable torque on all motors.

    Args:
        port: Serial port path (e.g., /dev/tty.usbmodem...)
        name: Human-readable name for logging

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"  Port: {port}")
    print(f"{'─' * 50}")

    try:
        bus = FeetechMotorsBus(port=port, motors=SO101_MOTORS)
        bus.connect(handshake=True)
        print("  ✓ Connected")

        bus.disable_torque()
        print("  ✓ Torque disabled on all motors")

        # Read current positions to confirm connection
        positions = bus.sync_read("Present_Position", normalize=False)
        print("  Motor positions (raw):")
        for motor, pos in positions.items():
            print(f"    {motor}: {pos}")

        bus.disconnect(disable_torque=False)  # Already disabled
        print("  ✓ Disconnected")
        return True

    except ConnectionError as e:
        print(f"  ✗ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Release (disable torque on) SO-101 robot motors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              # Release both follower and leader
  %(prog)s --follower   # Release only the follower arm
  %(prog)s --leader     # Release only the leader arm
  %(prog)s --port /dev/tty.usbmodem123  # Release a specific port
        """,
    )
    parser.add_argument(
        "--follower",
        "-f",
        action="store_true",
        help="Release only the follower arm",
    )
    parser.add_argument(
        "--leader",
        "-l",
        action="store_true",
        help="Release only the leader arm",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=str,
        help="Release motors on a specific port",
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  SO-101 Motor Release Utility")
    print("=" * 50)

    results = []

    if args.port:
        # Custom port specified
        results.append(release_motors(args.port, "Custom Port"))
    elif args.follower and not args.leader:
        # Only follower
        results.append(release_motors(ROBOT_PORT, "Follower Arm"))
    elif args.leader and not args.follower:
        # Only leader
        results.append(release_motors(LEADER_PORT, "Leader Arm"))
    else:
        # Both (default)
        results.append(release_motors(ROBOT_PORT, "Follower Arm"))
        results.append(release_motors(LEADER_PORT, "Leader Arm"))

    # Summary
    print("\n" + "=" * 50)
    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print(f"  ✓ All arms released successfully ({success_count}/{total_count})")
        print("  Arms can now be moved freely by hand.")
    else:
        print(f"  ⚠ Some arms failed to release ({success_count}/{total_count})")

    print("=" * 50 + "\n")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
