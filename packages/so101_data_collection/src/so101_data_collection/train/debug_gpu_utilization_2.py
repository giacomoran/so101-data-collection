#!/usr/bin/env python3
"""GPU utilization monitoring script for debugging LeRobot training performance.

This version waits for GPU usage to reach 20% before starting measurements,
to exclude the initialization phase from statistics.
"""

import argparse
import warnings

# Suppress pynvml deprecation warning (nvidia-ml-py provides the same module)
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")
import csv
import subprocess
import threading
import time
from pathlib import Path

import pynvml


def monitor_gpu(
    output_file: Path,
    stop_event: threading.Event,
    warmup_threshold: float = 20.0,
    interval: float = 0.5,
):
    """Monitor GPU utilization and write to CSV file.

    Args:
        output_file: Path to output CSV file
        stop_event: Event to signal monitoring should stop
        warmup_threshold: GPU utilization percentage to wait for before starting measurements
        interval: Monitoring interval in seconds
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Phase 1: Wait for GPU utilization to reach warmup_threshold
    print(f"Waiting for GPU utilization to reach {warmup_threshold}% before starting measurements...")
    warmup_reached = False
    while not stop_event.is_set() and not warmup_reached:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        if gpu_util >= warmup_threshold:
            warmup_reached = True
            print(f"GPU utilization reached {gpu_util}%, starting measurements.")
        else:
            time.sleep(interval)

    if stop_event.is_set():
        pynvml.nvmlShutdown()
        return

    # Phase 2: Record measurements
    start_time = time.time()

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed", "gpu_util", "mem_util", "mem_used_mb", "mem_total_mb"])

        while not stop_event.is_set():
            elapsed = time.time() - start_time

            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used // (1024 * 1024)
            mem_total_mb = mem_info.total // (1024 * 1024)

            writer.writerow([f"{elapsed:.1f}", gpu_util, mem_util, mem_used_mb, mem_total_mb])
            f.flush()

            time.sleep(interval)

    pynvml.nvmlShutdown()


def compute_stats(csv_file: Path) -> dict:
    """Compute summary statistics from GPU monitoring CSV.

    Args:
        csv_file: Path to CSV file with GPU monitoring data

    Returns:
        Dictionary with avg/min/max statistics
    """
    gpu_utils = []
    mem_utils = []
    mem_useds = []

    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu_utils.append(int(row["gpu_util"]))
            mem_utils.append(int(row["mem_util"]))
            mem_useds.append(int(row["mem_used_mb"]))

    if not gpu_utils:
        return {
            "avg_gpu_util": 0,
            "avg_mem_used": 0,
            "max_mem_used": 0,
        }

    return {
        "avg_gpu_util": sum(gpu_utils) / len(gpu_utils),
        "avg_mem_used": sum(mem_useds) / len(mem_useds),
        "max_mem_used": max(mem_useds),
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU utilization during training")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file path")
    parser.add_argument("--cmd", type=str, required=True, help="Training command to run")
    parser.add_argument(
        "--warmup-threshold",
        type=float,
        default=20.0,
        help="GPU utilization percentage to wait for before starting measurements (default: 20)",
    )
    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Start GPU monitoring in background thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(args.output, stop_event, args.warmup_threshold),
    )
    monitor_thread.start()

    print(f"Starting training with GPU monitoring...")
    print(f"Command: {args.cmd}")
    print(f"GPU stats will be saved to: {args.output}")
    print(f"Warmup threshold: {args.warmup_threshold}%")
    print("-" * 80)

    # Run training command
    try:
        result = subprocess.run(args.cmd, shell=True, check=True)
        exit_code = result.returncode
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print(f"\nTraining failed with exit code {exit_code}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        exit_code = 130
    finally:
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()

    # Compute and print statistics
    print("-" * 80)
    print("GPU Monitoring Summary (excluding initialization phase):")
    stats = compute_stats(args.output)
    print(f"  Average GPU utilization: {stats['avg_gpu_util']:.1f}%")
    print(f"  Average memory used: {stats['avg_mem_used']:.0f} MiB")
    print(f"  Peak memory used: {stats['max_mem_used']:.0f} MiB")
    print(f"\nFull stats saved to: {args.output}")

    return exit_code


if __name__ == "__main__":
    exit(main())
