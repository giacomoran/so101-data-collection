#!/usr/bin/env python3
"""
Homing FPS sweep experiment.

Runs at 120 FPS, always recording observations.
Sends actions only at the control rate (10, 20, 30, 60 Hz).

This measures how control frequency affects trajectory tracking.

Usage:
    python src/latency/sweep_homing_fps.py
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

# Add the package to the path
sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "packages/so101_data_collection/src"),
)

from so101_data_collection.shared.homing import (
    GRIPPER_OPEN,
    JOINT_ACTION_HOME,
    SAFE_HEIGHT,
    get_joint_positions,
)
from so101_data_collection.shared.setup import ROBOT_ID, ROBOT_PORT

# Configuration
FPS_OBSERVATION = 120  # Observation recording rate
LIST_FPS_CONTROL = [10, 20, 30, 60]  # Control FPS values to sweep
MAX_VELOCITY_DEG_PER_SEC = 50.0  # From homing.py
NAMES_JOINT = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class RecordingObservation:
    """Recorded observations during a homing run."""

    fps_control: int
    timestamps: list[float] = field(default_factory=list)
    positions: dict[str, list[float]] = field(default_factory=dict)  # Observed
    trajectory: dict[str, list[float]] = field(default_factory=dict)  # Interpolated target
    actions: dict[str, list[float]] = field(default_factory=dict)  # Last sent to robot
    is_control_frame: list[bool] = field(default_factory=list)
    timestamps_phase: list[float] = field(default_factory=list)  # Phase start times

    def __post_init__(self) -> None:
        for name_joint in NAMES_JOINT:
            self.positions[name_joint] = []
            self.trajectory[name_joint] = []
            self.actions[name_joint] = []


def compute_trajectory(
    start_positions: dict[str, float],
    target_positions: dict[str, float],
    elapsed: float,
) -> tuple[dict[str, float], bool]:
    """
    Compute interpolated positions for all joints.

    Returns:
        (positions, is_complete): Interpolated positions and whether trajectory is done
    """
    positions = {}
    is_complete = True

    for name_joint in NAMES_JOINT:
        start = start_positions.get(name_joint, 0.0)
        target = target_positions.get(name_joint, start)
        distance = abs(target - start)

        if distance < 0.1:  # Already at target
            positions[name_joint] = target
            continue

        duration = distance / MAX_VELOCITY_DEG_PER_SEC
        t = min(elapsed / duration, 1.0) if duration > 0 else 1.0

        positions[name_joint] = start + t * (target - start)

        if t < 1.0:
            is_complete = False

    return positions, is_complete


def run_trajectory_phase(
    robot: SO101Follower,
    target_positions: dict[str, float],
    fps_control: int,
    recording: RecordingObservation,
    time_base: float,
    last_commanded: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Run a single trajectory phase at 120 FPS observation, variable control rate.

    Args:
        robot: Connected robot
        target_positions: Target joint positions (keys with .pos suffix)
        fps_control: Control rate in Hz
        recording: Recording to append data to
        time_base: Base time for timestamps in recording
        last_commanded: Last commanded positions from previous phase (keys without .pos)
                       Used to maintain continuity for joints not moving in this phase.

    Returns:
        Final commanded positions (keys without .pos suffix)
    """
    period_obs = 1.0 / FPS_OBSERVATION
    control_divisor = FPS_OBSERVATION // fps_control

    # Get current observed position
    obs = robot.get_observation()
    obs_positions = get_joint_positions(obs)

    # If no previous commanded positions, use observed
    if last_commanded is None:
        last_commanded = obs_positions.copy()

    # Build start and target positions:
    # - Start from last commanded (for continuity)
    # - Target is either the new target (moving joints) or same as start (non-moving)
    start_positions = {}
    targets = {}
    for name_joint in NAMES_JOINT:
        # Always start from last commanded for continuity
        start_positions[name_joint] = last_commanded.get(name_joint, obs_positions.get(name_joint, 0.0))

        key = f"{name_joint}.pos"
        if key in target_positions:
            # Moving joint: interpolate to target
            targets[name_joint] = target_positions[key]
        else:
            # Non-moving joint: stay at start
            targets[name_joint] = start_positions[name_joint]

    time_start = time.perf_counter()
    idx_frame = 0

    # Mark phase start
    recording.timestamps_phase.append(time_start - time_base)

    # Track last sent action
    last_action_sent = {f"{j}.pos": start_positions[j] for j in NAMES_JOINT}

    while True:
        time_loop_start = time.perf_counter()
        elapsed = time_loop_start - time_start

        # Compute interpolated positions
        interp_positions, is_complete = compute_trajectory(start_positions, targets, elapsed)

        # Check if this is a control frame
        is_control = (idx_frame % control_divisor) == 0

        # Send action only on control frames
        if is_control:
            action = {f"{j}.pos": interp_positions[j] for j in NAMES_JOINT}
            robot.send_action(action)
            last_action_sent = action

        # Always read observation
        obs = robot.get_observation()
        obs_positions = get_joint_positions(obs)

        # Record
        recording.timestamps.append(time_loop_start - time_base)
        recording.is_control_frame.append(is_control)
        for name_joint in NAMES_JOINT:
            recording.positions[name_joint].append(obs_positions.get(name_joint, 0.0))
            recording.trajectory[name_joint].append(interp_positions[name_joint])
            recording.actions[name_joint].append(last_action_sent[f"{name_joint}.pos"])

        # Check completion
        if is_complete:
            action = {f"{j}.pos": targets[j] for j in NAMES_JOINT}
            robot.send_action(action)
            last_action_sent = action
            break

        idx_frame += 1

        # Maintain observation rate
        elapsed_loop = time.perf_counter() - time_loop_start
        if elapsed_loop < period_obs:
            precise_sleep(period_obs - elapsed_loop)

    # Return final commanded positions (without .pos suffix)
    return {j: last_action_sent[f"{j}.pos"] for j in NAMES_JOINT}


def run_homing_experiment(robot: SO101Follower, fps_control: int) -> RecordingObservation:
    """
    Run complete homing sequence at specified control rate.

    Phases:
    1. Lift to safe height (shoulder_pan, shoulder_lift, elbow_flex)
    2. Open gripper
    3. Move wrist joints and close gripper
    4. Lower to home position
    """
    recording = RecordingObservation(fps_control=fps_control)
    time_base = time.perf_counter()

    print("  Phase 1: Lift to safe height...")
    last_commanded = run_trajectory_phase(robot, SAFE_HEIGHT, fps_control, recording, time_base)

    print("  Phase 2: Open gripper...")
    last_commanded = run_trajectory_phase(
        robot, {"gripper.pos": GRIPPER_OPEN}, fps_control, recording, time_base, last_commanded
    )

    print("  Phase 3: Move wrist and close gripper...")
    wrist_targets = {
        "wrist_flex.pos": JOINT_ACTION_HOME["wrist_flex.pos"],
        "wrist_roll.pos": JOINT_ACTION_HOME["wrist_roll.pos"],
        "gripper.pos": JOINT_ACTION_HOME["gripper.pos"],
    }
    last_commanded = run_trajectory_phase(robot, wrist_targets, fps_control, recording, time_base, last_commanded)

    print("  Phase 4: Lower to home...")
    lower_targets = {
        "shoulder_lift.pos": JOINT_ACTION_HOME["shoulder_lift.pos"],
        "elbow_flex.pos": JOINT_ACTION_HOME["elbow_flex.pos"],
    }
    run_trajectory_phase(robot, lower_targets, fps_control, recording, time_base, last_commanded)

    return recording


def plot_recordings(recordings: list[RecordingObservation], path_output: Path) -> None:
    """Plot joint trajectories for all FPS values."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(recordings)))

    for idx_joint, name_joint in enumerate(NAMES_JOINT):
        ax = axes[idx_joint]
        key_joint = f"{name_joint}.pos"
        target = JOINT_ACTION_HOME.get(key_joint, 0.0)

        for recording, color in zip(recordings, colors):
            timestamps = np.array(recording.timestamps)
            positions = np.array(recording.positions[name_joint])

            ax.plot(
                timestamps,
                positions,
                color=color,
                label=f"{recording.fps_control} Hz",
                alpha=0.8,
                linewidth=1.0,
            )

        ax.axhline(y=target, color="red", linestyle="--", alpha=0.5, label="Target")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (deg)")
        ax.set_title(f"{name_joint}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Homing Trajectory @ {FPS_OBSERVATION} Hz obs, variable control rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(path_output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {path_output}")


def plot_tracking_error(recordings: list[RecordingObservation], path_output: Path) -> None:
    """Plot tracking error over time."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(recordings)))

    for idx_joint, name_joint in enumerate(NAMES_JOINT):
        ax = axes[idx_joint]
        key_joint = f"{name_joint}.pos"
        target = JOINT_ACTION_HOME.get(key_joint, 0.0)

        for recording, color in zip(recordings, colors):
            timestamps = np.array(recording.timestamps)
            positions = np.array(recording.positions[name_joint])
            errors = np.abs(positions - target)

            ax.plot(
                timestamps,
                errors,
                color=color,
                label=f"{recording.fps_control} Hz",
                alpha=0.8,
                linewidth=1.0,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (deg)")
        ax.set_title(f"{name_joint}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle(f"Tracking Error @ {FPS_OBSERVATION} Hz obs, variable control rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(path_output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {path_output}")


def plot_action_vs_observation(recordings: list[RecordingObservation], path_output: Path) -> None:
    """Plot trajectory, sent actions, and observed positions. One plot per FPS value."""
    colors_joint = plt.cm.tab10(np.linspace(0, 1, len(NAMES_JOINT)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx_fps, recording in enumerate(recordings):
        ax = axes[idx_fps]

        timestamps = np.array(recording.timestamps)
        control_frames = np.array(recording.is_control_frame)

        for idx_joint, name_joint in enumerate(NAMES_JOINT):
            color = colors_joint[idx_joint]
            positions = np.array(recording.positions[name_joint])
            trajectory = np.array(recording.trajectory[name_joint])
            actions = np.array(recording.actions[name_joint])

            # Plot trajectory (dashed)
            ax.plot(timestamps, trajectory, "--", color=color, alpha=0.5, linewidth=1.0)

            # Plot observed (solid)
            ax.plot(timestamps, positions, "-", color=color, alpha=0.8, linewidth=1.0, label=name_joint)

            # Plot sent actions as dots (only on control frames)
            ax.scatter(
                timestamps[control_frames],
                actions[control_frames],
                c=[color],
                s=6,
                alpha=0.4,
                zorder=5,
            )

        # Add vertical lines for phase boundaries
        for t_phase in recording.timestamps_phase:
            ax.axvline(x=t_phase, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (deg)")
        ax.set_title(f"Control @ {recording.fps_control} Hz")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Trajectory (dashed) vs Observed (solid) vs Sent (dots) @ {FPS_OBSERVATION} Hz obs", fontsize=12)
    plt.tight_layout()
    plt.savefig(path_output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {path_output}")


def main() -> int:
    """Run the FPS sweep experiment."""
    print("\n" + "=" * 60)
    print("  Homing FPS Sweep Experiment")
    print("=" * 60)
    print(f"\n  Observation rate: {FPS_OBSERVATION} Hz")
    print(f"  Control FPS values: {LIST_FPS_CONTROL}")
    print(f"\n  Port: {ROBOT_PORT}")
    print(f"  Robot ID: {ROBOT_ID}")

    # Setup robot
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        use_degrees=True,
        max_relative_target=None,
    )
    robot = SO101Follower(robot_config)

    print("\nConnecting to robot...")
    try:
        robot.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return 1

    if not robot.is_connected:
        print("Robot is not connected!")
        return 1

    print("Robot connected successfully")

    recordings: list[RecordingObservation] = []

    try:
        for idx, fps_control in enumerate(LIST_FPS_CONTROL):
            print(f"\n{'=' * 60}")
            print(f"  Run {idx + 1}/{len(LIST_FPS_CONTROL)}: Control FPS = {fps_control} Hz")
            print("=" * 60)

            if idx > 0:
                print("\nWaiting before next run...")
                time.sleep(2.0)

            recording = run_homing_experiment(robot, fps_control)
            recordings.append(recording)

            count_obs = len(recording.timestamps)
            count_control = sum(recording.is_control_frame)
            duration = recording.timestamps[-1] if recording.timestamps else 0
            print(f"  Recorded {count_obs} observations, {count_control} control frames")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Actual obs rate: {count_obs / duration:.1f} Hz" if duration > 0 else "")

    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        robot.disconnect()
        return 1

    print("\nDisconnecting robot...")
    robot.disconnect()

    print("\nGenerating plots...")
    path_project_root = Path(__file__).parent.parent.parent
    path_output_dir = path_project_root / "outputs" / "sweep_homing_fps"
    path_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_recordings(recordings, path_output_dir / f"trajectory_{timestamp}.png")
    plot_tracking_error(recordings, path_output_dir / f"error_{timestamp}.png")
    plot_action_vs_observation(recordings, path_output_dir / f"action_vs_obs_{timestamp}.png")

    print("\n" + "=" * 60)
    print("  Experiment complete")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
