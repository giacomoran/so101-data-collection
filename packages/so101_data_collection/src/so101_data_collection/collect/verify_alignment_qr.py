#!/usr/bin/env python
"""
QR code-based alignment verification for datasets.

This tool verifies temporal alignment by reading QR codes embedded in camera frames.
The QR codes encode timestamps, allowing direct measurement of:
- Mean camera latency (QR timestamp vs frame timestamp)
- Latency variance/jitter
- Frame timing consistency

Assumes the dataset episode was recorded while displaying a rolling QR code
(like in measure_camera_raw.py).

Usage:
    python -m so101_data_collection.collect.verify_alignment_qr \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided_1 \
        --episode 0

    # Specify camera to analyze
    python -m so101_data_collection.collect.verify_alignment_qr \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided_1 \
        --episode 0 \
        --camera wrist
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from so101_data_collection.collect.collect import (
    DEFAULT_DATASET_ROOT,
    MOTOR_NAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is in uint8 format for OpenCV."""
    if image.dtype == np.uint8:
        return image

    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        else:
            return image.astype(np.uint8)

    return image.astype(np.uint8)


def _to_numpy_hwc(img: Any) -> np.ndarray:
    """Convert image to numpy HWC format."""
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    # Handle torch.Tensor
    try:
        import torch

        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            return img_np
    except ImportError:
        pass

    # Try direct conversion
    return np.array(img)


def decode_qr_timestamp(
    image: np.ndarray, detector: cv2.QRCodeDetector
) -> float | None:
    """
    Decode timestamp from QR code in image.

    Returns timestamp as float, or None if QR code not found/decoded.
    """
    # Ensure proper format for OpenCV
    image = _ensure_uint8(image)

    # Convert to grayscale for QR detection (if needed)
    if len(image.shape) == 3:
        # OpenCV's QR detector works on color images, but grayscale can help
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Try decoding from grayscale
    data, _, _ = detector.detectAndDecode(gray)
    if data:
        try:
            return float(data)
        except ValueError:
            pass

    # Try color image
    if len(image.shape) == 3:
        data, _, _ = detector.detectAndDecode(image)
        if data:
            try:
                return float(data)
            except ValueError:
                pass

    return None


def load_dataset_episode(
    dataset_path: Path | None,
    repo_id: str,
    episode_idx: int = 0,
) -> dict[str, Any]:
    """Load a single episode from a dataset."""
    # Load metadata first
    try:
        if dataset_path and dataset_path.exists():
            meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)
        else:
            meta = LeRobotDatasetMetadata(repo_id)
    except Exception as e:
        raise RuntimeError(f"Could not load metadata for '{repo_id}': {e}")

    num_episodes = len(meta.episodes["dataset_from_index"])
    if episode_idx >= num_episodes:
        raise ValueError(
            f"Episode {episode_idx} not found. Dataset has {num_episodes} episodes."
        )

    logger.info(f"Loading episode {episode_idx} of {num_episodes}")

    # Load dataset with specific episode
    if dataset_path and dataset_path.exists():
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
            episodes=[episode_idx],
        )
    else:
        dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=[episode_idx],
        )

    # Load all frames
    num_frames = len(dataset)
    frames = []

    print(f"Loading episode {episode_idx} frames...", end="", flush=True)
    for idx in range(num_frames):
        frame = dataset[idx]
        frames.append(frame)
        if (idx + 1) % 10 == 0:
            print(
                f"\rLoading episode {episode_idx} frames... {idx + 1}/{num_frames}",
                end="",
                flush=True,
            )
    print(f"\rLoading episode {episode_idx} frames... {num_frames}/{num_frames}")

    return {
        "frames": frames,
        "fps": dataset.fps,
        "num_frames": num_frames,
    }


def verify_alignment_qr(
    repo_id: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    episode_idx: int = 0,
    camera_name: str = "wrist",
    output_path: Path | None = None,
) -> None:
    """
    Verify alignment using QR codes in camera frames.

    Args:
        repo_id: HuggingFace repo_id
        dataset_root: Root directory containing datasets
        episode_idx: Episode index to verify
        camera_name: Camera to analyze (wrist, top)
        output_path: Optional output path for plot
    """
    # Load dataset
    dataset_path = dataset_root / repo_id
    if dataset_path.exists():
        logger.info(f"Found local dataset at {dataset_path}")
    else:
        logger.info(f"Will stream from HuggingFace Hub")
        dataset_path = None

    data = load_dataset_episode(dataset_path, repo_id, episode_idx)

    frames = data["frames"]
    fps = data["fps"]
    num_frames = data["num_frames"]

    logger.info(f"Loaded {num_frames} frames at {fps} FPS")

    # Initialize QR decoder
    detector = cv2.QRCodeDetector()

    # Extract timestamps and decode QR codes
    image_key = f"observation.images.{camera_name}"

    qr_timestamps: list[float] = []
    frame_indices: list[int] = []
    frame_timestamps: list[float] = []  # Based on frame index and FPS
    latencies_ms: list[float] = []

    logger.info(f"Decoding QR codes from {camera_name} camera...")

    for idx, frame in enumerate(frames):
        if image_key not in frame:
            logger.warning(f"Camera {camera_name} not found in frame {idx}")
            continue

        # Get frame timestamp (index-based)
        frame_time = idx / fps
        frame_timestamps.append(frame_time)

        # Get image and decode QR
        img = frame[image_key]
        img_np = _to_numpy_hwc(img)

        qr_time = decode_qr_timestamp(img_np, detector)

        if qr_time is not None:
            qr_timestamps.append(qr_time)
            frame_indices.append(idx)

            # Calculate latency (we can't directly compare perf_counter times
            # across sessions, so we compute relative latency)
            # For this verification, we look at the pattern and variance

            if (idx + 1) % 20 == 0:
                print(
                    f"\rDecoding QR codes... {idx + 1}/{num_frames}", end="", flush=True
                )

    print(f"\rDecoding QR codes... {num_frames}/{num_frames}")

    if len(qr_timestamps) < 2:
        logger.error(
            f"Only found {len(qr_timestamps)} QR codes. Need at least 2 for analysis."
        )
        return

    logger.info(f"Successfully decoded {len(qr_timestamps)}/{num_frames} QR codes")

    # Convert to arrays
    qr_timestamps = np.array(qr_timestamps)
    frame_indices = np.array(frame_indices)

    # Calculate inter-frame intervals from QR timestamps
    qr_intervals = np.diff(qr_timestamps) * 1000  # ms
    expected_interval = 1000 / fps

    # Calculate latency as offset between QR time and expected time
    # Normalize by subtracting first QR timestamp
    qr_normalized = qr_timestamps - qr_timestamps[0]
    expected_times = frame_indices / fps
    latencies_ms = (expected_times - qr_normalized) * 1000

    # Statistics
    print("\n" + "=" * 60)
    print("QR CODE ALIGNMENT VERIFICATION")
    print("=" * 60)
    print(f"Dataset: {repo_id}")
    print(f"Episode: {episode_idx}")
    print(f"Camera: {camera_name}")
    print(f"Dataset FPS: {fps}Hz")
    print(f"QR codes decoded: {len(qr_timestamps)}/{num_frames}")
    print()

    print("Inter-frame Intervals (from QR timestamps):")
    print(f"  Expected:  {expected_interval:.1f}ms")
    print(f"  Mean:      {np.mean(qr_intervals):.1f}ms")
    print(f"  Std:       {np.std(qr_intervals):.1f}ms")
    print(f"  Min:       {np.min(qr_intervals):.1f}ms")
    print(f"  Max:       {np.max(qr_intervals):.1f}ms")
    print()

    print("Latency (frame time - QR time):")
    print(f"  Mean:      {np.mean(latencies_ms):.1f}ms")
    print(f"  Std:       {np.std(latencies_ms):.1f}ms")
    print(f"  Min:       {np.min(latencies_ms):.1f}ms")
    print(f"  Max:       {np.max(latencies_ms):.1f}ms")
    print(f"  P5:        {np.percentile(latencies_ms, 5):.1f}ms")
    print(f"  P95:       {np.percentile(latencies_ms, 95):.1f}ms")
    print("=" * 60)

    # Create visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Plot 1: Latency over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(frame_indices, latencies_ms, "b-", linewidth=1, alpha=0.7)
    ax1.axhline(
        y=np.mean(latencies_ms),
        color="r",
        linestyle="--",
        label=f"Mean ({np.mean(latencies_ms):.1f}ms)",
    )
    ax1.fill_between(
        frame_indices,
        np.mean(latencies_ms) - np.std(latencies_ms),
        np.mean(latencies_ms) + np.std(latencies_ms),
        alpha=0.2,
        color="r",
        label=f"±1 std ({np.std(latencies_ms):.1f}ms)",
    )
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Camera Latency Over Time (Frame Time - QR Time)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(latencies_ms, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(x=np.mean(latencies_ms), color="r", linestyle="--", label="Mean")
    ax2.axvline(x=np.median(latencies_ms), color="g", linestyle="--", label="Median")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Latency Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Inter-frame interval histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(qr_intervals, bins=30, edgecolor="black", alpha=0.7)
    ax3.axvline(
        x=expected_interval,
        color="r",
        linestyle="--",
        label=f"Expected ({expected_interval:.1f}ms)",
    )
    ax3.axvline(
        x=np.mean(qr_intervals),
        color="g",
        linestyle="--",
        label=f"Mean ({np.mean(qr_intervals):.1f}ms)",
    )
    ax3.set_xlabel("Interval (ms)")
    ax3.set_ylabel("Count")
    ax3.set_title("Inter-Frame Interval Distribution (from QR timestamps)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: QR timestamps vs expected timestamps
    ax4 = fig.add_subplot(gs[2, 0])
    expected_times_plot = frame_indices / fps
    qr_times_plot = qr_normalized
    ax4.plot(
        frame_indices,
        expected_times_plot,
        "b-",
        label="Expected (from frame index)",
        linewidth=1.5,
    )
    ax4.plot(
        frame_indices,
        qr_times_plot,
        "r--",
        label="QR timestamp (normalized)",
        linewidth=1.5,
    )
    ax4.set_xlabel("Frame Index")
    ax4.set_ylabel("Time (s)")
    ax4.set_title("Frame Timing: Expected vs QR")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Cumulative drift
    ax5 = fig.add_subplot(gs[2, 1])
    drift = (expected_times_plot - qr_times_plot) * 1000
    ax5.plot(frame_indices, drift, "b-", linewidth=1.5)
    ax5.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Frame Index")
    ax5.set_ylabel("Cumulative Drift (ms)")
    ax5.set_title("Cumulative Timing Drift")
    ax5.grid(True, alpha=0.3)

    plt.suptitle(
        f"QR Code Alignment Verification: {repo_id}\n"
        f"Episode {episode_idx}, Camera: {camera_name}",
        fontsize=12,
        fontweight="bold",
    )

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved verification plot to {output_path}")
    else:
        default_path = Path("alignment_verification_qr.png")
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved verification plot to {default_path}")

    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify dataset alignment using QR codes in camera frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo_id",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to verify",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="wrist",
        help="Camera to analyze (wrist, top)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for verification plot",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    verify_alignment_qr(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        episode_idx=args.episode,
        camera_name=args.camera,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
