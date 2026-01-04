#!/usr/bin/env python
"""
Measure execution latency for Feetech STS3215 servos using sinusoidal commands.

Following UMI paper methodology:
1. Send sinusoidal position commands to servos
2. Continuously read back actual positions
3. Compute optimal alignment via cross-correlation to find phase lag
4. l_action = l_e2e - l_obs (where l_obs comes from measure_proprioception.py)

The end-to-end latency (l_e2e) captures:
- Command send time
- Servo processing time
- Physical movement time
- Position readback time

Reports per-servo and aggregate statistics.

Usage:
    python -m src.latency.measure_execution --port /dev/tty.usbmodem...
    python -m src.latency.measure_execution --duration 10 --frequency 0.5
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so101_follower.config_so101_follower import (
    SO101FollowerConfig,
)
from scipy import signal

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.setup import ROBOT_ID, ROBOT_PORT  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Motor configuration for SO-101 arms
SO101_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

# STS3215 servo raw position range: 0-4095 (12-bit, 0-360 degrees)
# Center position: 2048 (180 degrees)
# Using raw values to avoid calibration requirement
SERVO_CENTER_RAW = 2048  # Middle of servo range
SERVO_AMPLITUDE_RAW = 200  # ±200 raw units ≈ ±17 degrees (safe range)


@dataclass
class SignalData:
    """Recorded signal data for a motor."""

    name: str
    timestamps: list[float] = field(default_factory=list)
    commanded: list[float] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)

    def add_sample(self, t: float, cmd: float, actual: float) -> None:
        self.timestamps.append(t)
        self.commanded.append(cmd)
        self.actual.append(actual)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.timestamps),
            np.array(self.commanded),
            np.array(self.actual),
        )


@dataclass
class LatencyResult:
    """Latency measurement result for a motor."""

    name: str
    e2e_latency_ms: float  # End-to-end latency from cross-correlation
    correlation: float  # Peak correlation value (quality indicator)
    samples: int
    sample_rate_hz: float


@dataclass
class MeasurementResult:
    """Results from execution latency measurement."""

    per_motor: dict[str, LatencyResult] = field(default_factory=dict)
    signal_data: dict[str, SignalData] = field(default_factory=dict)
    proprioception_latency_ms: float = 0.0  # From measure_proprioception.py or default


def compute_latency_cross_correlation(
    timestamps: np.ndarray,
    commanded: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    """
    Compute end-to-end latency using cross-correlation.

    The actual signal should lag behind the commanded signal by the e2e latency.

    Args:
        timestamps: Time array
        commanded: Commanded position signal
        actual: Actual position signal (read back)

    Returns:
        Tuple of (latency_ms, correlation_coefficient)
    """
    # Normalize signals (remove mean, scale to unit variance)
    cmd_norm = (commanded - np.mean(commanded)) / (np.std(commanded) + 1e-8)
    act_norm = (actual - np.mean(actual)) / (np.std(actual) + 1e-8)

    # Cross-correlation
    correlation = signal.correlate(act_norm, cmd_norm, mode="full")
    lags = signal.correlation_lags(len(act_norm), len(cmd_norm), mode="full")

    # Find peak correlation (should be positive lag = actual lags behind commanded)
    peak_idx = np.argmax(correlation)
    peak_lag = lags[peak_idx]
    peak_corr = correlation[peak_idx] / len(cmd_norm)  # Normalize

    # Convert lag to time
    dt = np.mean(np.diff(timestamps))
    latency_s = peak_lag * dt
    latency_ms = latency_s * 1000

    return latency_ms, peak_corr


def get_motor_center_position(
    bus: FeetechMotorsBus, motor_name: str, robot_id: str | None = None
) -> int:
    """
    Get the center position for a motor (middle of joint range).

    Uses calibration if robot_id is provided, otherwise falls back to servo center.

    Args:
        bus: Connected motor bus (should have calibration loaded if robot_id provided)
        motor_name: Name of motor
        robot_id: Robot ID for loading calibration

    Returns:
        Center position in raw servo units
    """
    # Try to use calibration if available
    if robot_id and hasattr(bus, "calibration") and bus.calibration is not None:
        if motor_name in bus.calibration:
            cal = bus.calibration[motor_name]
            # Get min and max from calibration
            min_pos = cal.min_position
            max_pos = cal.max_position
            center_raw = int((min_pos + max_pos) / 2)
            logger.debug(
                f"{motor_name}: calibration center = {center_raw} "
                f"(range: {min_pos}-{max_pos})"
            )
            return center_raw

    # Fallback to servo center if no calibration
    logger.warning(
        f"{motor_name}: No calibration available, using servo center ({SERVO_CENTER_RAW})"
    )
    return SERVO_CENTER_RAW


def move_all_motors_to_center(
    bus: FeetechMotorsBus,
    robot_id: str | None = None,
    motors: list[str] | None = None,
    speed_raw_per_sec: float = 100.0,
    update_rate_hz: float = 20.0,
) -> None:
    """
    Move all motors to their center positions SLOWLY so user can stop if needed.

    Moves at a controlled speed so the user has time to press Ctrl+C if the arm
    is about to hit something.

    Args:
        bus: Connected motor bus
        robot_id: Robot ID for loading calibration
        motors: List of motors to move (None = all motors)
        speed_raw_per_sec: Maximum speed in raw units per second (~100 = slow, safe)
        update_rate_hz: How often to update positions
    """
    if motors is None:
        motors = list(SO101_MOTORS.keys())

    logger.info("=" * 60)
    logger.info("MOVING ALL MOTORS TO CENTER POSITIONS")
    logger.info("  ⚠️  Press Ctrl+C to stop if arm is about to hit something!")
    logger.info("=" * 60)

    # Get center (target) positions for all motors
    target_positions = {}
    for motor_name in motors:
        target_positions[motor_name] = get_motor_center_position(
            bus, motor_name, robot_id
        )
        logger.info(f"  {motor_name}: target = {target_positions[motor_name]}")

    # Enable torque on all motors
    bus.enable_torque(motors=motors)
    time.sleep(0.1)

    # Read current positions
    current_positions = bus.sync_read(
        "Present_Position", motors=motors, normalize=False
    )
    logger.info("Current positions:")
    for motor_name in motors:
        diff = target_positions[motor_name] - current_positions[motor_name]
        logger.info(
            f"  {motor_name}: {current_positions[motor_name]} (need to move {diff:+d})"
        )

    # Calculate step size based on speed and update rate
    step_size = speed_raw_per_sec / update_rate_hz
    update_interval = 1.0 / update_rate_hz

    logger.info(f"Moving at ~{speed_raw_per_sec:.0f} raw units/sec...")
    logger.info("  (Press Ctrl+C to abort)")

    # Move incrementally towards center
    positions = {k: float(v) for k, v in current_positions.items()}
    max_iterations = 1000  # Safety limit

    for iteration in range(max_iterations):
        # Calculate next positions (move towards target at controlled speed)
        all_at_target = True
        for motor_name in motors:
            current = positions[motor_name]
            target = target_positions[motor_name]
            distance = target - current

            # If we're close enough, snap to target
            if abs(distance) <= step_size:
                positions[motor_name] = float(target)
            else:
                # Move one step towards target
                direction = 1 if distance > 0 else -1
                positions[motor_name] = current + direction * step_size
                all_at_target = False

        # Send new positions (convert to int for servo)
        int_positions = {k: int(v) for k, v in positions.items()}
        bus.sync_write("Goal_Position", int_positions, normalize=False)

        # Check if we've reached all targets
        if all_at_target:
            logger.info("✓ All motors reached center positions")
            break

        # Progress log every second
        if iteration % int(update_rate_hz) == 0 and iteration > 0:
            # Show progress for each motor
            progress_str = ", ".join(
                f"{m}: {int(positions[m])}"
                for m in motors[:3]  # Show first 3
            )
            if len(motors) > 3:
                progress_str += ", ..."
            logger.info(f"  Moving... [{progress_str}]")

        # Wait before next update
        time.sleep(update_interval)

    else:
        logger.warning("Movement timed out - may not have reached all targets")

    # Final wait to ensure motors have settled
    time.sleep(0.5)
    logger.info("Motors settled at center positions")


def run_sinusoidal_test(
    bus: FeetechMotorsBus,
    motor_name: str,
    robot_id: str | None = None,
    duration_s: float = 10.0,
    frequency_hz: float = 0.5,
    amplitude_raw: int = SERVO_AMPLITUDE_RAW,
    sample_rate_hz: float = 50.0,
) -> SignalData:
    """
    Run sinusoidal test on a single motor using raw servo values.

    Sends sinusoidal position commands and records actual positions.
    Uses raw values (no normalization) to avoid calibration requirement.

    Args:
        bus: Connected motor bus
        motor_name: Name of motor to test
        robot_id: Robot ID for calibration (used to get joint center)
        duration_s: Test duration in seconds
        frequency_hz: Sinusoid frequency
        amplitude_raw: Sinusoid amplitude in raw servo units
        sample_rate_hz: Sampling rate for recording

    Returns:
        SignalData with timestamps, commanded, and actual positions
    """
    data = SignalData(name=motor_name)
    dt = 1.0 / sample_rate_hz

    # Get center position for this motor (from calibration if available)
    center_raw = get_motor_center_position(bus, motor_name, robot_id)

    # Ensure torque is enabled for this motor (should already be enabled from move_all_motors_to_center)
    bus.enable_torque(motors=[motor_name])
    time.sleep(0.1)

    # Ensure motor is at center position (should already be there, but verify)
    logger.info(
        f"Ensuring {motor_name} is at center position ({center_raw} raw units)..."
    )
    bus.sync_write("Goal_Position", {motor_name: center_raw}, normalize=False)
    time.sleep(1.0)  # Brief wait to ensure position

    logger.info(f"Starting sinusoidal test for {motor_name}")
    logger.info(
        f"  Frequency: {frequency_hz} Hz, Amplitude: ±{amplitude_raw} raw units, "
        f"Duration: {duration_s}s"
    )

    start_time = time.perf_counter()
    last_sample_time = start_time

    try:
        while True:
            current_time = time.perf_counter()
            elapsed = current_time - start_time

            if elapsed >= duration_s:
                break

            # Generate sinusoidal command in raw units
            cmd_raw = center_raw + int(
                amplitude_raw * np.sin(2 * np.pi * frequency_hz * elapsed)
            )

            # Send command (raw values, no normalization)
            bus.sync_write("Goal_Position", {motor_name: cmd_raw}, normalize=False)

            # Read actual position (raw values)
            positions = bus.sync_read(
                "Present_Position", motors=[motor_name], normalize=False
            )
            actual_raw = positions[motor_name]

            # Record sample
            data.add_sample(elapsed, float(cmd_raw), float(actual_raw))

            # Maintain sample rate
            time_since_last = current_time - last_sample_time
            if time_since_last < dt:
                time.sleep(dt - time_since_last)
            last_sample_time = time.perf_counter()

    finally:
        # Return to center (keep torque enabled for next motor test)
        logger.info(f"Returning {motor_name} to center position...")
        bus.sync_write("Goal_Position", {motor_name: center_raw}, normalize=False)
        time.sleep(1.0)
        # Don't disable torque here - we'll disable all at the end

    return data


def run_measurement(
    port: str,
    robot_id: str | None = None,
    duration_s: float = 10.0,
    frequency_hz: float = 0.5,
    amplitude_raw: int = SERVO_AMPLITUDE_RAW,
    sample_rate_hz: float = 50.0,
    motors_to_test: list[str] | None = None,
    proprioception_latency_ms: float = 0.0,
) -> MeasurementResult:
    """
    Run execution latency measurement.

    Args:
        port: Serial port path
        robot_id: Robot ID for loading calibration (used to get joint center positions)
        duration_s: Test duration per motor
        frequency_hz: Sinusoid frequency
        amplitude_raw: Sinusoid amplitude in raw servo units
        sample_rate_hz: Sampling rate
        motors_to_test: List of motors to test (None = all)
        proprioception_latency_ms: Known proprioception latency to subtract

    Returns:
        MeasurementResult with per-motor latencies
    """
    logger.info(f"Connecting to motor bus on {port}...")

    # Load calibration if robot_id is provided
    bus = FeetechMotorsBus(port=port, motors=SO101_MOTORS)

    if robot_id:
        logger.info(f"Loading calibration for robot ID: {robot_id}")
        try:
            # Use robot config to load calibration
            # The config creates a motors_bus internally with calibration loaded
            robot_config = SO101FollowerConfig(port=port, id=robot_id)

            # Access the calibrated bus from the config
            # Note: The config may need to be initialized/connected first
            if hasattr(robot_config, "_motors_bus") or hasattr(
                robot_config, "motors_bus"
            ):
                # Try to get the motors_bus (it may be lazy-loaded)
                calibrated_bus = getattr(robot_config, "motors_bus", None) or getattr(
                    robot_config, "_motors_bus", None
                )

                if calibrated_bus is None:
                    # Motors bus might be created lazily, try creating robot instance
                    from lerobot.robots.so101_follower import SO101Follower

                    temp_robot = SO101Follower(robot_config)
                    temp_robot.connect()
                    calibrated_bus = temp_robot.motors_bus
                    temp_robot.disconnect()

                if (
                    calibrated_bus
                    and hasattr(calibrated_bus, "calibration")
                    and calibrated_bus.calibration
                ):
                    bus.calibration = calibrated_bus.calibration
                    logger.info("Calibration loaded successfully")
                else:
                    logger.warning("Robot config bus has no calibration")
            else:
                logger.warning(
                    f"Could not access motors_bus from config for {robot_id}"
                )
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}, using servo center")
            import traceback

            logger.debug(traceback.format_exc())

    bus.connect(handshake=True)
    logger.info("Connected")

    result = MeasurementResult(proprioception_latency_ms=proprioception_latency_ms)

    if motors_to_test is None:
        motors_to_test = list(SO101_MOTORS.keys())

    # Move ALL motors to center positions first to avoid collisions
    move_all_motors_to_center(bus, robot_id=robot_id, motors=motors_to_test)

    try:
        for motor_name in motors_to_test:
            logger.info(f"\nTesting motor: {motor_name}")

            # Run sinusoidal test
            signal_data = run_sinusoidal_test(
                bus=bus,
                motor_name=motor_name,
                robot_id=robot_id,
                duration_s=duration_s,
                frequency_hz=frequency_hz,
                amplitude_raw=amplitude_raw,
                sample_rate_hz=sample_rate_hz,
            )

            result.signal_data[motor_name] = signal_data

            # Compute latency
            timestamps, commanded, actual = signal_data.as_arrays()

            if len(timestamps) < 10:
                logger.warning(f"  Too few samples for {motor_name}, skipping")
                continue

            e2e_latency_ms, correlation = compute_latency_cross_correlation(
                timestamps, commanded, actual
            )

            actual_sample_rate = len(timestamps) / (timestamps[-1] - timestamps[0])

            result.per_motor[motor_name] = LatencyResult(
                name=motor_name,
                e2e_latency_ms=e2e_latency_ms,
                correlation=correlation,
                samples=len(timestamps),
                sample_rate_hz=actual_sample_rate,
            )

            logger.info(
                f"  {motor_name}: e2e={e2e_latency_ms:.1f}ms, "
                f"correlation={correlation:.3f}, "
                f"samples={len(timestamps)}"
            )

            # Brief pause between motors
            time.sleep(0.5)

    finally:
        # Return all motors to center and disable torque
        logger.info("\nReturning all motors to center positions...")
        if motors_to_test:
            center_positions = {
                motor: get_motor_center_position(bus, motor, robot_id)
                for motor in motors_to_test
            }
            bus.sync_write("Goal_Position", center_positions, normalize=False)
            time.sleep(2.0)

        bus.disable_torque()
        bus.disconnect()
        logger.info("Disconnected")

    return result


def print_results(result: MeasurementResult) -> None:
    """Print measurement results to console."""
    print("\n" + "=" * 80)
    print("EXECUTION LATENCY MEASUREMENT RESULTS")
    print("=" * 80)
    print("Method: Sinusoidal command alignment via cross-correlation")
    print(f"Proprioception latency (l_obs): {result.proprioception_latency_ms:.2f} ms")
    print("-" * 80)

    if not result.per_motor:
        print("No measurements recorded.")
        return

    print(
        f"{'Motor':<15} {'E2E (ms)':>12} {'l_action (ms)':>14} {'Correlation':>12} "
        f"{'Samples':>10} {'Rate (Hz)':>10}"
    )
    print("-" * 80)

    e2e_values = []
    action_values = []

    for motor_name, lat_result in result.per_motor.items():
        action_latency = lat_result.e2e_latency_ms - result.proprioception_latency_ms

        e2e_values.append(lat_result.e2e_latency_ms)
        action_values.append(action_latency)

        print(
            f"{motor_name:<15} "
            f"{lat_result.e2e_latency_ms:>11.2f}ms "
            f"{action_latency:>13.2f}ms "
            f"{lat_result.correlation:>12.3f} "
            f"{lat_result.samples:>10} "
            f"{lat_result.sample_rate_hz:>9.1f}"
        )

    # Aggregate statistics
    print("-" * 80)
    print(
        f"{'AGGREGATE':<15} "
        f"{np.mean(e2e_values):>11.2f}ms "
        f"{np.mean(action_values):>13.2f}ms "
        f"{'':>12} "
        f"{'':>10} "
        f"{'':>10}"
    )
    print(
        f"{'(std)':<15} {np.std(e2e_values):>11.2f}ms {np.std(action_values):>13.2f}ms"
    )

    print("=" * 80)
    print("\nLegend:")
    print("  E2E:        End-to-end latency (command send → position read)")
    print("  l_action:   Execution latency = E2E - l_obs (proprioception latency)")
    print("  Correlation: Cross-correlation peak (>0.8 is good, >0.9 is excellent)")
    print("=" * 80)


def save_csv(result: MeasurementResult, path: Path) -> None:
    """Save signal data to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["motor", "timestamp", "commanded", "actual"])

        for motor_name, signal_data in result.signal_data.items():
            for t, cmd, act in zip(
                signal_data.timestamps, signal_data.commanded, signal_data.actual
            ):
                writer.writerow([motor_name, t, cmd, act])

    logger.info(f"Saved signal data to {path}")


def save_summary_csv(result: MeasurementResult, path: Path) -> None:
    """Save summary results to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "motor",
                "e2e_latency_ms",
                "action_latency_ms",
                "correlation",
                "samples",
                "sample_rate_hz",
            ]
        )

        for motor_name, lat_result in result.per_motor.items():
            action_latency = (
                lat_result.e2e_latency_ms - result.proprioception_latency_ms
            )
            writer.writerow(
                [
                    motor_name,
                    lat_result.e2e_latency_ms,
                    action_latency,
                    lat_result.correlation,
                    lat_result.samples,
                    lat_result.sample_rate_hz,
                ]
            )

    logger.info(f"Saved summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure execution latency for Feetech STS3215 servos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=str,
        default=ROBOT_PORT,
        help="Serial port for motor bus",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration per motor in seconds",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=0.5,
        help="Sinusoid frequency in Hz",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default=ROBOT_ID,
        help="Robot ID (for future calibration support)",
    )
    parser.add_argument(
        "--amplitude",
        type=int,
        default=SERVO_AMPLITUDE_RAW,
        help="Sinusoid amplitude in raw servo units (default: ±200 ≈ ±17 degrees)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=50.0,
        help="Sampling rate in Hz",
    )
    parser.add_argument(
        "--motors",
        type=str,
        nargs="+",
        choices=list(SO101_MOTORS.keys()),
        default=None,
        help="Motors to test (default: all)",
    )
    parser.add_argument(
        "--proprioception-latency",
        type=float,
        default=0.0,
        help="Known proprioception latency in ms (from measure_proprioception.py)",
    )
    parser.add_argument(
        "--save-signals",
        type=Path,
        default=None,
        help="Path to save raw signal data as CSV",
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
        help="Path to save summary results as CSV",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Execution Latency Measurement (Sinusoidal Method)")
    logger.info("=" * 60)
    logger.info(f"Port: {args.port}")
    logger.info(f"Robot ID: {args.robot_id}")
    logger.info(f"Duration: {args.duration}s per motor")
    logger.info(f"Frequency: {args.frequency} Hz")
    logger.info(f"Amplitude: ±{args.amplitude} raw units")
    logger.info(f"Sample rate: {args.sample_rate} Hz")
    logger.info(f"Proprioception latency: {args.proprioception_latency} ms")

    print("\n⚠️  WARNING: This test will move the servos!")
    print("    Make sure the arm has clearance and won't hit anything.")
    print("    Press Ctrl+C to abort.\n")
    time.sleep(2)

    result = run_measurement(
        port=args.port,
        robot_id=args.robot_id,
        duration_s=args.duration,
        frequency_hz=args.frequency,
        amplitude_raw=args.amplitude,
        sample_rate_hz=args.sample_rate,
        motors_to_test=args.motors,
        proprioception_latency_ms=args.proprioception_latency,
    )

    print_results(result)

    if args.save_signals:
        save_csv(result, args.save_signals)

    if args.save_summary:
        save_summary_csv(result, args.save_summary)


if __name__ == "__main__":
    main()
