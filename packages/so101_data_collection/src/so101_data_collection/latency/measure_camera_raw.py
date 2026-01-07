#!/usr/bin/env python
"""
Raw camera latency measurement - bypasses lerobot to find minimum achievable latency.

Tests multiple capture strategies:
1. Default OpenCV capture (baseline - matches lerobot behavior)
2. OpenCV with CAP_PROP_BUFFERSIZE=1 (attempt to disable buffering)
3. OpenCV with frame flushing (grab and discard old frames)
4. OpenCV with MJPG codec (often lower latency than YUYV)

The rolling QR code display is automatically started as a separate process.

Usage:
    python -m src.latency.measure_camera_raw --camera 1 --duration 10
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from so101_data_collection.shared.setup import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    WRIST_CAMERA_INDEX,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptureMethod(str, Enum):
    DEFAULT = "default"  # Standard OpenCV capture (baseline)
    BUFFER_1 = "buffer_1"  # CAP_PROP_BUFFERSIZE = 1
    FLUSH = "flush"  # Read and discard frames before measurement
    MJPG = "mjpg"  # Use MJPG codec
    MJPG_FLUSH = "mjpg_flush"  # MJPG + flush
    GRAB_ONLY = "grab_only"  # Use grab() to skip decoding until needed


@dataclass
class Stats:
    """Latency statistics."""

    latencies_ms: list[float] = field(default_factory=list)
    failed: int = 0
    total: int = 0

    def add(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.total += 1

    def add_failed(self) -> None:
        self.failed += 1
        self.total += 1

    def summary(self) -> dict:
        if not self.latencies_ms:
            return {"count": 0, "failed": self.failed}
        arr = np.array(self.latencies_ms)
        return {
            "count": len(self.latencies_ms),
            "failed": self.failed,
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
        }


def setup_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    method: CaptureMethod,
) -> cv2.VideoCapture:
    """
    Setup camera with specific capture method.

    Returns configured VideoCapture object.
    """
    # Try different backends on macOS
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            logger.info(f"Opened camera with backend: {backend}")
            break
        cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}")

    # Set resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Apply method-specific settings
    if method == CaptureMethod.BUFFER_1:
        # Try to set buffer size to 1 (not always supported)
        result = cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"Set CAP_PROP_BUFFERSIZE=1: {result}")
        actual = cap.get(cv2.CAP_PROP_BUFFERSIZE)
        logger.info(f"Actual buffer size: {actual}")

    if method in (CaptureMethod.MJPG, CaptureMethod.MJPG_FLUSH):
        # Use MJPG codec (often lower latency)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        result = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        logger.info(f"Set MJPG codec: {result}")

    # Log actual settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
    logger.info(
        f"Camera settings: {actual_width}x{actual_height} @ {actual_fps}fps, codec={fourcc_str}"
    )

    return cap


def flush_buffer(cap: cv2.VideoCapture, num_frames: int = 5) -> None:
    """Read and discard frames to flush the buffer."""
    for _ in range(num_frames):
        cap.grab()


def read_frame(cap: cv2.VideoCapture, method: CaptureMethod) -> np.ndarray | None:
    """
    Read a frame using the specified method.

    Returns frame or None if read failed.
    """
    if method == CaptureMethod.GRAB_ONLY:
        # Use grab() to advance buffer, then retrieve() only when needed
        # This can help skip old frames
        for _ in range(3):  # Grab a few times to get latest
            cap.grab()
        ret, frame = cap.retrieve()
    elif method in (CaptureMethod.FLUSH, CaptureMethod.MJPG_FLUSH):
        # Flush buffer before reading
        flush_buffer(cap, num_frames=3)
        ret, frame = cap.read()
    else:
        # Standard read
        ret, frame = cap.read()

    return frame if ret else None


def decode_qr(image: np.ndarray, detector: cv2.QRCodeDetector) -> str | None:
    """Decode QR code from image."""
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None


def start_rolling_qr_process(fps: float = 30.0) -> subprocess.Popen:
    """
    Start the rolling QR code display as a separate process.

    The QR display will run indefinitely until terminated.

    Args:
        fps: Frame rate for QR display

    Returns:
        Popen process object for the QR display
    """
    cmd = [
        sys.executable,
        "-m",
        "so101_data_collection.shared.rolling_qr",
        "--fps",
        str(fps),
        # No --duration flag - runs indefinitely until terminated
    ]
    logger.info(f"Starting rolling QR display process: {' '.join(cmd)}")
    # Don't redirect stdout/stderr - GUI windows need direct access to the display
    # on macOS. Redirecting them prevents the window from appearing.
    # Let stdout/stderr go to the terminal so the GUI can work properly.
    process = subprocess.Popen(cmd)
    # Give it a moment to initialize and create the window
    time.sleep(1.0)
    if process.poll() is not None:
        # Process exited early - try to get error info
        returncode = process.returncode
        raise RuntimeError(
            f"QR display process exited early with return code {returncode}. "
            f"Check that OpenCV can create windows and that no other QR display is running."
        )
    logger.info("Rolling QR display started successfully")
    return process


def run_measurement(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    duration_s: float,
    method: CaptureMethod,
    display_latency_ms: float = 0.0,
    qr_process: subprocess.Popen | None = None,
) -> Stats:
    """Run latency measurement with specified capture method."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing method: {method.value}")
    logger.info(f"{'=' * 60}")

    cap = setup_camera(camera_index, width, height, fps, method)
    detector = cv2.QRCodeDetector()
    stats = Stats()

    try:
        # Warm up camera (this may include blocking operations like flushing)
        logger.info("Warming up camera...")
        for _ in range(30):
            cap.read()
        time.sleep(0.5)

        logger.info(f"Running measurement for {duration_s}s...")
        start_time = time.perf_counter()
        last_log = start_time

        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed >= duration_s:
                break

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Capture frame
            frame = read_frame(cap, method)
            # Record time AFTER reading frame (not before!) because flush methods
            # take time and we want the time when the frame is actually available
            # Use time.time() (wall-clock) to match QR display timestamps across processes
            t_recv = time.time()

            if frame is None:
                stats.add_failed()
                continue

            # Decode QR
            decoded = decode_qr(frame, detector)
            if decoded:
                try:
                    t_qr = float(decoded)
                    latency_ms = (t_recv - t_qr) * 1000 - display_latency_ms
                    stats.add(latency_ms)
                except ValueError:
                    stats.add_failed()
            else:
                stats.add_failed()

            # Progress log
            if loop_start - last_log >= 2.0:
                summary = stats.summary()
                if summary["count"] > 0:
                    logger.info(
                        f"[{elapsed:.1f}s] n={summary['count']}, "
                        f"mean={summary['mean_ms']:.1f}ms, "
                        f"min={summary['min_ms']:.1f}ms"
                    )
                last_log = loop_start

            # Maintain frame rate
            frame_time = time.perf_counter() - loop_start
            target = 1.0 / fps
            if frame_time < target:
                time.sleep(target - frame_time)

    finally:
        cap.release()
        # Note: QR process cleanup is handled by caller

    return stats


def print_comparison(results: dict[CaptureMethod, Stats]) -> None:
    """Print comparison table of all methods."""
    print("\n" + "=" * 80)
    print("COMPARISON OF CAPTURE METHODS")
    print("=" * 80)
    print(
        f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10} {'P95':>10}"
    )
    print("-" * 80)

    for method, stats in results.items():
        summary = stats.summary()
        if summary["count"] > 0:
            print(
                f"{method.value:<20} "
                f"{summary['mean_ms']:>9.1f}ms "
                f"{summary['std_ms']:>9.1f}ms "
                f"{summary['min_ms']:>9.1f}ms "
                f"{summary['max_ms']:>9.1f}ms "
                f"{summary['median_ms']:>9.1f}ms "
                f"{summary['p95_ms']:>9.1f}ms"
            )
        else:
            print(f"{method.value:<20} {'(no data)':<60}")

    print("=" * 80)

    # Find best method
    best_method = None
    best_latency = float("inf")
    for method, stats in results.items():
        summary = stats.summary()
        if summary["count"] > 0 and summary["mean_ms"] < best_latency:
            best_latency = summary["mean_ms"]
            best_method = method

    if best_method:
        print(f"\nBest method: {best_method.value} ({best_latency:.1f}ms mean latency)")

        # Compare to default
        default_summary = results.get(CaptureMethod.DEFAULT, Stats()).summary()
        if default_summary.get("count", 0) > 0:
            improvement = default_summary["mean_ms"] - best_latency
            print(
                f"Improvement over default: {improvement:.1f}ms "
                f"({improvement / default_summary['mean_ms'] * 100:.1f}%)"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raw camera latency measurement - test multiple capture methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=WRIST_CAMERA_INDEX,
        help="Camera index",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=CAMERA_WIDTH,
        help="Capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=CAMERA_HEIGHT,
        help="Capture height",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration per method in seconds",
    )
    parser.add_argument(
        "--display-latency",
        type=float,
        default=0.0,
        help="Display latency to subtract (ms)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=[m.value for m in CaptureMethod],
        default=[m.value for m in CaptureMethod],
        help="Methods to test",
    )
    parser.add_argument(
        "--single",
        type=str,
        choices=[m.value for m in CaptureMethod],
        help="Test only a single method (for quick tests)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which methods to test
    if args.single:
        methods = [CaptureMethod(args.single)]
    else:
        methods = [CaptureMethod(m) for m in args.methods]

    logger.info(f"Testing {len(methods)} capture method(s)")
    logger.info(
        f"Camera: {args.camera}, Resolution: {args.width}x{args.height} @ {args.fps}fps"
    )
    logger.info(f"Duration per method: {args.duration}s")

    # Start rolling QR display process (runs indefinitely until terminated)
    qr_process = None
    try:
        qr_process = start_rolling_qr_process(fps=args.fps)
    except Exception as e:
        logger.error(f"Failed to start rolling QR display: {e}")
        raise

    results: dict[CaptureMethod, Stats] = {}

    try:
        for method in methods:
            try:
                stats = run_measurement(
                    camera_index=args.camera,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    duration_s=args.duration,
                    method=method,
                    display_latency_ms=args.display_latency,
                    qr_process=qr_process,
                )
                results[method] = stats

                # Print individual results
                summary = stats.summary()
                print(f"\n{method.value}: ", end="")
                if summary["count"] > 0:
                    print(
                        f"mean={summary['mean_ms']:.1f}ms, "
                        f"min={summary['min_ms']:.1f}ms, "
                        f"max={summary['max_ms']:.1f}ms"
                    )
                else:
                    print("No successful measurements")

                # Brief pause between methods
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Method {method.value} failed: {e}")
                results[method] = Stats()

        # Print comparison
        if len(results) > 1:
            print_comparison(results)
        elif len(results) == 1:
            method, stats = next(iter(results.items()))
            summary = stats.summary()
            print("\n" + "=" * 50)
            print(f"RESULTS: {method.value}")
            print("=" * 50)
            if summary["count"] > 0:
                print(f"  Samples:  {summary['count']}")
                print(f"  Failed:   {summary['failed']}")
                print(f"  Mean:     {summary['mean_ms']:.2f}ms")
                print(f"  Std:      {summary['std_ms']:.2f}ms")
                print(f"  Min:      {summary['min_ms']:.2f}ms")
                print(f"  Max:      {summary['max_ms']:.2f}ms")
                print(f"  Median:   {summary['median_ms']:.2f}ms")
                print(f"  P95:      {summary['p95_ms']:.2f}ms")
    finally:
        # Cleanup QR process
        if qr_process is not None:
            logger.info("Stopping rolling QR display process...")
            qr_process.terminate()
            try:
                qr_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.warning("QR process didn't terminate, killing...")
                qr_process.kill()
                qr_process.wait()


if __name__ == "__main__":
    main()
