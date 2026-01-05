#!/usr/bin/env python
"""
Measure proprioception latency for Feetech STS3215 servos.

Since the servos don't support hardware timestamps, we measure round-trip time (RTT)
of read commands and approximate proprioception latency as RTT / 2.

Methodology (following UMI paper approach for non-timestamped hardware):
- l_obs ≈ RTT / 2 (assuming symmetric send/receive delays)

Reports per-servo and aggregate statistics.

Usage:
    python -m src.latency.measure_proprioception --port /dev/tty.usbmodem...
    python -m src.latency.measure_proprioception --duration 10 --samples 1000
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

# Add project root to Python path for imports
from so101_data_collection.shared.setup import ROBOT_ID, ROBOT_PORT

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


@dataclass
class LatencyStats:
    """Latency statistics for a single motor or aggregate."""

    name: str
    rtt_ms: list[float] = field(default_factory=list)

    def add(self, rtt_ms: float) -> None:
        self.rtt_ms.append(rtt_ms)

    @property
    def latency_ms(self) -> list[float]:
        """Proprioception latency ≈ RTT / 2."""
        return [rtt / 2 for rtt in self.rtt_ms]

    def summary(self) -> dict:
        if not self.rtt_ms:
            return {"name": self.name, "count": 0}

        rtt_arr = np.array(self.rtt_ms)
        lat_arr = rtt_arr / 2

        return {
            "name": self.name,
            "count": len(self.rtt_ms),
            "rtt_mean_ms": float(np.mean(rtt_arr)),
            "rtt_std_ms": float(np.std(rtt_arr)),
            "rtt_min_ms": float(np.min(rtt_arr)),
            "rtt_max_ms": float(np.max(rtt_arr)),
            "rtt_median_ms": float(np.median(rtt_arr)),
            "rtt_p95_ms": float(np.percentile(rtt_arr, 95)),
            "latency_mean_ms": float(np.mean(lat_arr)),
            "latency_std_ms": float(np.std(lat_arr)),
            "latency_min_ms": float(np.min(lat_arr)),
            "latency_max_ms": float(np.max(lat_arr)),
            "latency_median_ms": float(np.median(lat_arr)),
            "latency_p95_ms": float(np.percentile(lat_arr, 95)),
        }


@dataclass
class MeasurementResult:
    """Results from proprioception latency measurement."""

    per_motor: dict[str, LatencyStats] = field(default_factory=dict)
    aggregate: LatencyStats = field(default_factory=lambda: LatencyStats("aggregate"))
    sync_read_all: LatencyStats = field(
        default_factory=lambda: LatencyStats("sync_read_all")
    )


def measure_individual_motor_rtt(
    bus: FeetechMotorsBus,
    motor_name: str,
    num_samples: int,
    delay_between_ms: float = 1.0,
) -> LatencyStats:
    """
    Measure RTT for reading a single motor's position.

    Args:
        bus: Connected motor bus
        motor_name: Name of motor to read
        num_samples: Number of samples to collect
        delay_between_ms: Delay between samples in ms

    Returns:
        LatencyStats with RTT measurements
    """
    stats = LatencyStats(name=motor_name)

    for _ in range(num_samples):
        t_start = time.perf_counter()
        _ = bus.sync_read("Present_Position", motors=[motor_name], normalize=False)
        t_end = time.perf_counter()

        rtt_ms = (t_end - t_start) * 1000
        stats.add(rtt_ms)

        if delay_between_ms > 0:
            time.sleep(delay_between_ms / 1000)

    return stats


def measure_sync_read_all_rtt(
    bus: FeetechMotorsBus,
    num_samples: int,
    delay_between_ms: float = 1.0,
) -> LatencyStats:
    """
    Measure RTT for reading all motors at once (sync_read).

    This represents the actual proprioception latency in practice,
    since we typically read all motors together.

    Args:
        bus: Connected motor bus
        num_samples: Number of samples to collect
        delay_between_ms: Delay between samples in ms

    Returns:
        LatencyStats with RTT measurements
    """
    stats = LatencyStats(name="sync_read_all")

    for _ in range(num_samples):
        t_start = time.perf_counter()
        _ = bus.sync_read("Present_Position", normalize=False)
        t_end = time.perf_counter()

        rtt_ms = (t_end - t_start) * 1000
        stats.add(rtt_ms)

        if delay_between_ms > 0:
            time.sleep(delay_between_ms / 1000)

    return stats


def run_measurement(
    port: str,
    robot_id: str | None = None,
    num_samples: int = 500,
    delay_between_ms: float = 1.0,
    measure_individual: bool = True,
) -> MeasurementResult:
    """
    Run proprioception latency measurement.

    Args:
        port: Serial port path
        robot_id: Robot ID (for future calibration support, currently unused)
        num_samples: Number of samples per motor
        delay_between_ms: Delay between samples
        measure_individual: Whether to measure each motor individually

    Returns:
        MeasurementResult with per-motor and aggregate stats
    """
    logger.info(f"Connecting to motor bus on {port}...")
    if robot_id:
        logger.info(f"Robot ID: {robot_id} (calibration not yet used)")

    bus = FeetechMotorsBus(port=port, motors=SO101_MOTORS)
    bus.connect(handshake=True)
    logger.info("Connected")

    result = MeasurementResult()

    try:
        # Warm up - a few reads to stabilize
        logger.info("Warming up...")
        for _ in range(10):
            bus.sync_read("Present_Position", normalize=False)
        time.sleep(0.1)

        # Measure sync_read for all motors (most realistic scenario)
        logger.info(f"Measuring sync_read (all motors) - {num_samples} samples...")
        result.sync_read_all = measure_sync_read_all_rtt(
            bus, num_samples, delay_between_ms
        )
        summary = result.sync_read_all.summary()
        logger.info(
            f"  sync_read_all: RTT={summary['rtt_mean_ms']:.2f}ms, "
            f"latency={summary['latency_mean_ms']:.2f}ms"
        )

        # Measure individual motors
        if measure_individual:
            motor_names = list(SO101_MOTORS.keys())
            for motor_name in motor_names:
                logger.info(f"Measuring {motor_name} - {num_samples} samples...")
                stats = measure_individual_motor_rtt(
                    bus, motor_name, num_samples, delay_between_ms
                )
                result.per_motor[motor_name] = stats

                # Add to aggregate
                for rtt in stats.rtt_ms:
                    result.aggregate.add(rtt)

                summary = stats.summary()
                logger.info(
                    f"  {motor_name}: RTT={summary['rtt_mean_ms']:.2f}ms, "
                    f"latency={summary['latency_mean_ms']:.2f}ms"
                )

    finally:
        bus.disconnect()
        logger.info("Disconnected")

    return result


def print_results(result: MeasurementResult) -> None:
    """Print measurement results to console."""
    print("\n" + "=" * 70)
    print("PROPRIOCEPTION LATENCY MEASUREMENT RESULTS")
    print("=" * 70)
    print("Method: RTT / 2 (assuming symmetric serial delays)")
    print("-" * 70)

    # Sync read all (most important)
    print("\n📊 SYNC_READ ALL MOTORS (realistic scenario)")
    print("-" * 70)
    summary = result.sync_read_all.summary()
    if summary["count"] > 0:
        print(f"  Samples:           {summary['count']}")
        print(f"  RTT Mean:          {summary['rtt_mean_ms']:.3f} ms")
        print(f"  RTT Std:           {summary['rtt_std_ms']:.3f} ms")
        print(f"  RTT Min:           {summary['rtt_min_ms']:.3f} ms")
        print(f"  RTT Max:           {summary['rtt_max_ms']:.3f} ms")
        print(f"  RTT Median:        {summary['rtt_median_ms']:.3f} ms")
        print(f"  RTT P95:           {summary['rtt_p95_ms']:.3f} ms")
        print()
        print(f"  Latency Mean:      {summary['latency_mean_ms']:.3f} ms  ← l_obs")
        print(f"  Latency P95:       {summary['latency_p95_ms']:.3f} ms")

    # Per-motor results
    if result.per_motor:
        print("\n📊 PER-MOTOR RESULTS")
        print("-" * 70)
        print(
            f"{'Motor':<15} {'RTT Mean':>10} {'RTT Std':>10} {'RTT P95':>10} "
            f"{'Lat Mean':>10} {'Lat P95':>10}"
        )
        print("-" * 70)

        for motor_name, stats in result.per_motor.items():
            summary = stats.summary()
            if summary["count"] > 0:
                print(
                    f"{motor_name:<15} "
                    f"{summary['rtt_mean_ms']:>9.2f}ms "
                    f"{summary['rtt_std_ms']:>9.2f}ms "
                    f"{summary['rtt_p95_ms']:>9.2f}ms "
                    f"{summary['latency_mean_ms']:>9.2f}ms "
                    f"{summary['latency_p95_ms']:>9.2f}ms"
                )

        # Aggregate
        print("-" * 70)
        agg_summary = result.aggregate.summary()
        if agg_summary["count"] > 0:
            print(
                f"{'AGGREGATE':<15} "
                f"{agg_summary['rtt_mean_ms']:>9.2f}ms "
                f"{agg_summary['rtt_std_ms']:>9.2f}ms "
                f"{agg_summary['rtt_p95_ms']:>9.2f}ms "
                f"{agg_summary['latency_mean_ms']:>9.2f}ms "
                f"{agg_summary['latency_p95_ms']:>9.2f}ms"
            )

    print("=" * 70)


def save_csv(result: MeasurementResult, path: Path) -> None:
    """Save results to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["motor", "sample_idx", "rtt_ms", "latency_ms"])

        # Sync read all
        for i, rtt in enumerate(result.sync_read_all.rtt_ms):
            writer.writerow(["sync_read_all", i, rtt, rtt / 2])

        # Per motor
        for motor_name, stats in result.per_motor.items():
            for i, rtt in enumerate(stats.rtt_ms):
                writer.writerow([motor_name, i, rtt, rtt / 2])

    logger.info(f"Saved results to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure proprioception latency for Feetech STS3215 servos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=str,
        default=ROBOT_PORT,
        help="Serial port for motor bus",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default=ROBOT_ID,
        help="Robot ID (for future calibration support)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples per measurement",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between samples in ms",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Skip individual motor measurements (only measure sync_read all)",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Path to save raw measurements as CSV",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Proprioception Latency Measurement")
    logger.info("=" * 60)
    logger.info(f"Port: {args.port}")
    logger.info(f"Robot ID: {args.robot_id}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Delay between samples: {args.delay}ms")

    result = run_measurement(
        port=args.port,
        robot_id=args.robot_id,
        num_samples=args.samples,
        delay_between_ms=args.delay,
        measure_individual=not args.no_individual,
    )

    print_results(result)

    if args.save_csv:
        save_csv(result, args.save_csv)


if __name__ == "__main__":
    main()
