#!/usr/bin/env python
"""
Camera latency measurement script following UMI paper methodology.

Displays a rolling QR code with the current timestamp on screen, captures it
with the camera, and calculates end-to-end latency by decoding the QR code.

Latency formula: l_camera = t_recv - t_display - l_display

Where:
- t_recv: timestamp when frame is received from camera
- t_display: timestamp encoded in QR code (when it was generated)
- l_display: known monitor refresh latency (configurable)

Usage:
    python -m src.latency.measure_camera --camera 1 --duration 10
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import qrcode
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.setup import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    WRIST_CAMERA_INDEX,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# QR code display settings
QR_WINDOW_NAME = "Latency Measurement - QR Code"
QR_SIZE = 800  # Size of QR code display in pixels


@dataclass
class MeasureConfig:
    """Configuration for latency measurement."""

    camera_index: int = WRIST_CAMERA_INDEX
    fps: int = 30
    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT
    duration_s: float = 10.0
    display_latency_ms: float = 0.0  # Known monitor latency to subtract
    save_csv: Path | None = None
    save_video: Path | None = None


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    t_display: float  # Timestamp encoded in QR code
    t_recv: float  # Timestamp when frame was received
    latency_ms: float  # Calculated latency in ms


@dataclass
class LatencyStats:
    """Statistics from latency measurements."""

    measurements: list[LatencyMeasurement] = field(default_factory=list)
    failed_detections: int = 0
    total_frames: int = 0

    def add_measurement(self, m: LatencyMeasurement) -> None:
        self.measurements.append(m)
        self.total_frames += 1

    def add_failed_detection(self) -> None:
        self.failed_detections += 1
        self.total_frames += 1

    @property
    def latencies_ms(self) -> list[float]:
        return [m.latency_ms for m in self.measurements]

    def summary(self) -> dict:
        """Calculate summary statistics."""
        if not self.measurements:
            return {
                "count": 0,
                "failed": self.failed_detections,
                "total_frames": self.total_frames,
            }

        latencies = np.array(self.latencies_ms)
        return {
            "count": len(self.measurements),
            "failed": self.failed_detections,
            "total_frames": self.total_frames,
            "detection_rate": len(self.measurements) / self.total_frames * 100
            if self.total_frames > 0
            else 0,
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }


def generate_qr_image(data: str, size: int = QR_SIZE) -> np.ndarray:
    """
    Generate a QR code image as a numpy array.

    Args:
        data: String data to encode
        size: Output image size in pixels

    Returns:
        BGR image as numpy array
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Create PIL image
    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to numpy array
    img_array = np.array(img.convert("RGB"))

    # Resize to desired size
    img_resized = cv2.resize(img_array, (size, size), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    return img_bgr


def create_display_frame(timestamp: float, qr_size: int = QR_SIZE) -> np.ndarray:
    """
    Create a display frame with QR code encoding the timestamp.

    The frame includes:
    - Large QR code with timestamp
    - Text showing the encoded timestamp (for debugging)
    """
    # Generate QR code with timestamp
    timestamp_str = f"{timestamp:.6f}"
    qr_img = generate_qr_image(timestamp_str, qr_size)

    # Create frame with padding for text
    padding = 60
    frame_h = qr_size + padding
    frame_w = qr_size
    frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255

    # Place QR code
    frame[:qr_size, :qr_size] = qr_img

    # Add timestamp text below QR code
    text = f"t={timestamp_str}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame_w - text_size[0]) // 2
    text_y = qr_size + 35
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return frame


def decode_qr(
    image: np.ndarray, detector: cv2.QRCodeDetector
) -> tuple[str | None, np.ndarray | None]:
    """
    Decode QR code from image.

    Returns:
        Tuple of (decoded_data, points) or (None, None) if detection failed
    """
    data, points, _ = detector.detectAndDecode(image)
    if data:
        return data, points
    return None, None


def measure_latency(config: MeasureConfig) -> LatencyStats:
    """
    Run latency measurement.

    Displays QR codes with timestamps and measures camera capture latency.
    """
    logger.info(f"Starting latency measurement for {config.duration_s}s")
    logger.info(f"Camera index: {config.camera_index}, FPS: {config.fps}")
    logger.info(f"Display latency compensation: {config.display_latency_ms}ms")

    # Initialize camera
    cam_config = OpenCVCameraConfig(
        index_or_path=config.camera_index,
        fps=config.fps,
        width=config.width,
        height=config.height,
    )
    camera = OpenCVCamera(cam_config)

    # Initialize QR detector
    qr_detector = cv2.QRCodeDetector()

    # Stats tracking
    stats = LatencyStats()

    # Video writer for debug output
    video_writer = None
    if config.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(config.save_video),
            fourcc,
            config.fps,
            (config.width, config.height),
        )
        logger.info(f"Recording debug video to: {config.save_video}")

    # Create display window
    cv2.namedWindow(QR_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(QR_WINDOW_NAME, QR_SIZE, QR_SIZE + 60)

    try:
        # Connect camera
        camera.connect()
        logger.info("Camera connected, starting measurement...")
        logger.info("Point camera at the QR code window on your monitor")
        logger.info("Press 'q' to stop early")

        # Warm-up: display initial QR and wait for camera to stabilize
        warmup_frame = create_display_frame(time.perf_counter())
        cv2.imshow(QR_WINDOW_NAME, warmup_frame)
        cv2.waitKey(1)
        time.sleep(0.5)  # Let camera auto-exposure settle

        start_time = time.perf_counter()
        last_log_time = start_time
        target_frame_time = 1.0 / config.fps

        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed >= config.duration_s:
                break

            # Generate and display QR code with current timestamp
            t_display = time.perf_counter()
            display_frame = create_display_frame(t_display)
            cv2.imshow(QR_WINDOW_NAME, display_frame)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Measurement stopped by user")
                break

            # Capture frame from camera (same pattern as collect.py)
            t_recv = time.perf_counter()
            frame = camera.async_read()

            if frame is None:
                logger.warning("Camera returned None frame")
                stats.add_failed_detection()
                continue

            # Try to decode QR code
            decoded_data, points = decode_qr(frame, qr_detector)

            if decoded_data:
                try:
                    t_qr = float(decoded_data)
                    # Calculate latency: t_recv - t_display - l_display
                    latency_ms = (t_recv - t_qr) * 1000 - config.display_latency_ms

                    measurement = LatencyMeasurement(
                        t_display=t_qr,
                        t_recv=t_recv,
                        latency_ms=latency_ms,
                    )
                    stats.add_measurement(measurement)

                    # Draw detection on frame for video
                    if video_writer and points is not None:
                        points = points.astype(int)
                        for i in range(4):
                            pt1 = tuple(points[0][i])
                            pt2 = tuple(points[0][(i + 1) % 4])
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                        # Add latency text
                        cv2.putText(
                            frame,
                            f"Latency: {latency_ms:.1f}ms",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                except ValueError:
                    logger.warning(f"Failed to parse QR data: {decoded_data}")
                    stats.add_failed_detection()
            else:
                stats.add_failed_detection()
                if video_writer:
                    cv2.putText(
                        frame,
                        "No QR detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Write frame to video
            if video_writer:
                video_writer.write(frame)

            # Log progress every 2 seconds
            if loop_start - last_log_time >= 2.0:
                summary = stats.summary()
                if summary["count"] > 0:
                    logger.info(
                        f"[{elapsed:.1f}s] Samples: {summary['count']}, "
                        f"Mean: {summary['mean_ms']:.1f}ms, "
                        f"Detection rate: {summary['detection_rate']:.0f}%"
                    )
                else:
                    logger.info(
                        f"[{elapsed:.1f}s] No successful detections yet "
                        f"({stats.failed_detections} failed)"
                    )
                last_log_time = loop_start

            # Maintain frame rate
            frame_time = time.perf_counter() - loop_start
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)

    finally:
        # Cleanup
        camera.disconnect()
        cv2.destroyAllWindows()
        if video_writer:
            video_writer.release()

    return stats


def print_results(stats: LatencyStats, config: MeasureConfig) -> None:
    """Print measurement results to console."""
    summary = stats.summary()

    print("\n" + "=" * 50)
    print("CAMERA LATENCY MEASUREMENT RESULTS")
    print("=" * 50)
    print(f"Camera index: {config.camera_index}")
    print(f"Target FPS: {config.fps}")
    print(f"Duration: {config.duration_s}s")
    print(f"Display latency compensation: {config.display_latency_ms}ms")
    print("-" * 50)
    print(f"Total frames captured: {summary['total_frames']}")
    print(f"Successful QR detections: {summary['count']}")
    print(f"Failed detections: {summary['failed']}")
    print(f"Detection rate: {summary.get('detection_rate', 0):.1f}%")

    if summary["count"] > 0:
        print("-" * 50)
        print("LATENCY STATISTICS (ms)")
        print("-" * 50)
        print(f"  Mean:   {summary['mean_ms']:.2f}")
        print(f"  Std:    {summary['std_ms']:.2f}")
        print(f"  Min:    {summary['min_ms']:.2f}")
        print(f"  Max:    {summary['max_ms']:.2f}")
        print(f"  Median: {summary['median_ms']:.2f}")
        print(f"  P95:    {summary['p95_ms']:.2f}")
        print(f"  P99:    {summary['p99_ms']:.2f}")
    else:
        print("-" * 50)
        print("No successful measurements. Check that:")
        print("  - Camera is pointed at the QR code on screen")
        print("  - QR code is in focus and well-lit")
        print("  - Camera index is correct")

    print("=" * 50)


def save_csv(stats: LatencyStats, path: Path) -> None:
    """Save raw measurements to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_display", "t_recv", "latency_ms"])
        for m in stats.measurements:
            writer.writerow([m.t_display, m.t_recv, m.latency_ms])
    logger.info(f"Saved {len(stats.measurements)} measurements to {path}")


def print_histogram(stats: LatencyStats) -> None:
    """Print ASCII histogram of latency distribution."""
    if not stats.measurements:
        return

    latencies = stats.latencies_ms
    min_lat = min(latencies)
    max_lat = max(latencies)

    # Create bins
    n_bins = 20
    bin_width = (max_lat - min_lat) / n_bins if max_lat > min_lat else 1
    bins = [0] * n_bins

    for lat in latencies:
        bin_idx = min(int((lat - min_lat) / bin_width), n_bins - 1)
        bins[bin_idx] += 1

    max_count = max(bins)
    bar_width = 40

    print("\nLATENCY DISTRIBUTION")
    print("-" * 60)
    for i, count in enumerate(bins):
        bin_start = min_lat + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{bin_start:6.1f}-{bin_end:6.1f}ms | {bar} ({count})")


def parse_args() -> MeasureConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Camera latency measurement using QR codes (UMI paper method)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=WRIST_CAMERA_INDEX,
        help="Camera index to test",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target capture FPS",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=CAMERA_WIDTH,
        help="Camera capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=CAMERA_HEIGHT,
        help="Camera capture height",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Measurement duration in seconds",
    )
    parser.add_argument(
        "--display-latency",
        type=float,
        default=0.0,
        help="Known monitor display latency in ms (subtracted from measurements)",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Path to save raw measurements as CSV",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Path to save debug video showing QR detection",
    )

    args = parser.parse_args()

    return MeasureConfig(
        camera_index=args.camera,
        fps=args.fps,
        width=args.width,
        height=args.height,
        duration_s=args.duration,
        display_latency_ms=args.display_latency,
        save_csv=args.save_csv,
        save_video=args.save_video,
    )


def main() -> None:
    config = parse_args()
    stats = measure_latency(config)

    print_results(stats, config)
    print_histogram(stats)

    if config.save_csv:
        save_csv(stats, config.save_csv)


if __name__ == "__main__":
    main()
