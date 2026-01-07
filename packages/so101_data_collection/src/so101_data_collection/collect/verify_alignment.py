#!/usr/bin/env python
"""
Verification tool for dataset alignment.

This tool visualizes the temporal alignment between camera frames and proprioception
readings in a recorded dataset. High correlation between optical flow (camera motion)
and proprioception velocity indicates good alignment.

Usage:
    python -m so101_data_collection.collect.verify_alignment \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided_1 \
        --episode 0
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

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from so101_data_collection.collect.collect import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    MOTOR_NAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_episode(
    dataset_path: Path | None, repo_id: str, episode_idx: int = 0
) -> dict[str, Any]:
    """
    Load a single episode from a dataset (only downloads the needed episode's files).

    Args:
        dataset_path: Optional local path to dataset. If None, downloads from HuggingFace Hub.
        repo_id: Full HuggingFace repo_id (e.g., 'giacomoran/so101_data_collection_cube_hand_guided_1')
        episode_idx: Index of episode to load

    Returns:
        Dictionary with:
        - 'frames': list of frame dicts with observation.state, observation.images.*, action
        - 'fps': dataset FPS
        - 'num_frames': number of frames
    """

    # Load metadata first (lightweight, only downloads meta/ folder)
    try:
        if dataset_path and dataset_path.exists():
            meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)
        else:
            meta = LeRobotDatasetMetadata(repo_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not load metadata for repo_id '{repo_id}'. "
            f"If dataset is local, provide --dataset-root. Error: {e}"
        )

    # Get number of episodes from the episodes dataframe
    num_episodes = len(meta.episodes["dataset_from_index"])
    if episode_idx >= num_episodes:
        raise ValueError(
            f"Episode {episode_idx} not found. Dataset has {num_episodes} episodes."
        )

    logger.info(f"Loading episode {episode_idx} of {num_episodes}")

    # Load dataset with episodes=[episode_idx] to only download/load the needed episode
    # LeRobotDataset.download() uses get_episodes_file_paths() which returns only
    # the specific data and video files needed for the requested episodes
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

    # Load all frames from the episode with progress
    num_frames = len(dataset)
    frames = []

    print(f"Loading episode {episode_idx} frames...", end="", flush=True)

    for idx in range(num_frames):
        frame = dataset[idx]
        frames.append(frame)
        # Update progress every 10 frames or on first frame
        if (idx + 1) % 10 == 0 or idx == 0:
            print(
                f"\rLoading episode {episode_idx} frames... {idx + 1}/{num_frames}",
                end="",
                flush=True,
            )

    # Print final count on new line
    print(f"\rLoading episode {episode_idx} frames... {num_frames}/{num_frames}")

    return {
        "frames": frames,
        "fps": dataset.fps,
        "num_frames": num_frames,
    }


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in uint8 format for OpenCV optical flow.

    Handles:
    - float32/float64 images in [0, 1] range -> scale to [0, 255]
    - float32/float64 images in [0, 255] range -> convert directly
    - uint8 images -> pass through
    """
    if image.dtype == np.uint8:
        return image

    # Float image - check range
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            # Normalized [0, 1] range - scale up
            return (image * 255).astype(np.uint8)
        else:
            # Already in [0, 255] range
            return image.astype(np.uint8)

    # Other dtypes - try to convert
    return image.astype(np.uint8)


def compute_optical_flow(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute optical flow magnitude between two images.

    Returns:
        Mean magnitude of optical flow vectors.
    """
    # Ensure uint8 format for Farneback algorithm
    image1 = _ensure_uint8(image1)
    image2 = _ensure_uint8(image2)

    gray1 = (
        cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
    )
    gray2 = (
        cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2
    )

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(magnitude))


def compute_proprioception_velocity(
    state1: np.ndarray, state2: np.ndarray, dt: float
) -> float:
    """
    Compute velocity magnitude from proprioception states.

    Returns:
        Mean velocity magnitude across all joints.
    """
    if dt <= 0:
        return 0.0
    velocity = np.abs((state2 - state1) / dt)
    return float(np.mean(velocity))


def verify_alignment(
    repo_id: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    episode_idx: int = 0,
    output_path: Path | None = None,
) -> None:
    """
    Verify temporal alignment between camera frames and proprioception in a dataset.

    Args:
        repo_id: Full HuggingFace repo_id (e.g., 'giacomoran/so101_data_collection_cube_hand_guided_1')
        dataset_root: Root directory containing datasets (checks repo_id as path within this)
        episode_idx: Episode index to verify
        output_path: Optional output path for the plot

    Creates visualization plots showing:
    1. Proprioception states over time (joint positions)
    2. Proprioception velocity (rate of change)
    3. Camera motion via optical flow
    4. Motion correlation (optical flow vs proprioception velocity)
    5. Frame timing consistency

    High correlation between optical flow and proprioception velocity indicates
    good temporal alignment between camera and proprioception data.
    """
    # Check if dataset exists locally, otherwise will stream from HuggingFace Hub
    # Try repo_id as-is (e.g., giacomoran/so101_data_collection_cube_hand_guided_1)
    dataset_path = dataset_root / repo_id
    if dataset_path.exists():
        logger.info(f"Found local dataset at {dataset_path}")
    else:
        logger.info(
            f"Local dataset not found at {dataset_path}, will stream from HuggingFace Hub"
        )
        dataset_path = None  # Signal to stream from Hub

    data = load_dataset_episode(dataset_path, repo_id, episode_idx)

    frames = data["frames"]
    fps = data["fps"]
    num_frames = data["num_frames"]

    logger.info(f"Loaded {num_frames} frames at {fps} FPS")

    # Extract proprioception states
    proprioception_states = []
    timestamps = []

    for i, frame in enumerate(frames):
        # Extract state array
        if isinstance(frame["observation.state"], dict):
            # Convert dict to array
            state_array = np.array(
                [frame["observation.state"][f"{motor}.pos"] for motor in MOTOR_NAMES],
                dtype=np.float32,
            )
        else:
            state_array = np.array(frame["observation.state"], dtype=np.float32)

        proprioception_states.append(state_array)

        # Get timestamp (either from frame or compute from index)
        if "timestamp" in frame:
            timestamps.append(frame["timestamp"])
        else:
            timestamps.append(i / fps)

    proprioception_states = np.array(proprioception_states)
    timestamps = np.array(timestamps)

    # Compute velocities
    proprioception_velocities = []
    for i in range(1, len(proprioception_states)):
        dt = timestamps[i] - timestamps[i - 1]
        vel = compute_proprioception_velocity(
            proprioception_states[i - 1], proprioception_states[i], dt
        )
        proprioception_velocities.append(vel)
    proprioception_velocities = np.array(proprioception_velocities)

    # Compute optical flow for camera images
    camera_flows = {}
    # Find available camera keys
    available_cameras = set()
    for frame in frames:
        for key in frame.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                available_cameras.add(cam_name)

    for cam_name in available_cameras:
        flows = []
        images = []

        # Collect images
        for frame in frames:
            image_key = f"observation.images.{cam_name}"
            if image_key in frame:
                img = frame[image_key]
                # Handle different image formats
                if isinstance(img, np.ndarray):
                    images.append(img)
                else:
                    # Try to convert to numpy array (handles torch.Tensor)
                    try:
                        import torch

                        if isinstance(img, torch.Tensor):
                            img_np = img.cpu().numpy()
                            # Handle channel ordering: torch uses CHW, OpenCV expects HWC
                            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                                img_np = np.transpose(img_np, (1, 2, 0))
                            images.append(img_np)
                        elif hasattr(img, "numpy"):
                            img_np = img.numpy()
                            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                                img_np = np.transpose(img_np, (1, 2, 0))
                            images.append(img_np)
                        else:
                            # Try direct conversion
                            img_np = np.array(img)
                            images.append(img_np)
                    except Exception as e:
                        logger.warning(
                            f"Failed to convert image type {type(img)} for {image_key}: {e}"
                        )

        # Compute flow between consecutive frames
        for i in range(1, len(images)):
            try:
                flow = compute_optical_flow(images[i - 1], images[i])
                flows.append(flow)
            except Exception as e:
                logger.warning(f"Optical flow computation failed for {cam_name}: {e}")
                flows.append(0.0)

        if flows:
            camera_flows[cam_name] = np.array(flows)

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Proprioception states over time
    ax1 = fig.add_subplot(gs[0, :])
    for i, motor_name in enumerate(MOTOR_NAMES):
        ax1.plot(
            timestamps,
            proprioception_states[:, i],
            label=motor_name,
            alpha=0.7,
            linewidth=1,
        )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Joint Position")
    ax1.set_title("Proprioception States Over Time")
    ax1.legend(ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Proprioception velocity
    ax2 = fig.add_subplot(gs[1, 0])
    velocity_timestamps = timestamps[1:]
    ax2.plot(velocity_timestamps, proprioception_velocities, "b-", linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mean Velocity Magnitude")
    ax2.set_title("Proprioception Velocity")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Optical flow (if available)
    ax3 = fig.add_subplot(gs[1, 1])
    for cam_name, flows in camera_flows.items():
        flow_timestamps = timestamps[1 : len(flows) + 1]
        ax3.plot(
            flow_timestamps,
            flows,
            label=f"{cam_name} camera",
            linewidth=1.5,
            alpha=0.8,
        )
    if camera_flows:
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Optical Flow Magnitude")
        ax3.set_title("Camera Motion (Optical Flow)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No camera data available", ha="center", va="center")
        ax3.set_title("Camera Motion (Optical Flow)")

    # Plot 4: Correlation between proprioception velocity and optical flow
    ax4 = fig.add_subplot(gs[2, :])
    if camera_flows:
        # Align timestamps for correlation
        min_len = min(
            len(proprioception_velocities), min(len(f) for f in camera_flows.values())
        )
        prop_vel_aligned = proprioception_velocities[:min_len]

        # Normalize proprioception velocity (same for all cameras)
        prop_norm = (prop_vel_aligned - prop_vel_aligned.min()) / (
            prop_vel_aligned.max() - prop_vel_aligned.min() + 1e-8
        )

        # Plot proprioception velocity once (not per camera)
        ax4.plot(
            velocity_timestamps[:min_len],
            prop_norm,
            label="Proprioception velocity (norm)",
            linewidth=1.5,
            alpha=0.7,
            color="blue",
        )

        # Plot optical flow for each camera
        colors = ["orange", "green", "red", "purple"]  # Colors for different cameras
        for i, (cam_name, flows) in enumerate(camera_flows.items()):
            flow_aligned = flows[:min_len]
            flow_norm = (flow_aligned - flow_aligned.min()) / (
                flow_aligned.max() - flow_aligned.min() + 1e-8
            )

            correlation = np.corrcoef(prop_norm, flow_norm)[0, 1]

            ax4.plot(
                velocity_timestamps[:min_len],
                flow_norm,
                label=f"{cam_name} optical flow (norm, corr={correlation:.3f})",
                linewidth=1.5,
                alpha=0.7,
                color=colors[i % len(colors)],
            )

        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Normalized Magnitude")
        ax4.set_title("Motion Correlation: Proprioception vs Camera")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No camera data for correlation", ha="center", va="center")
        ax4.set_title("Motion Correlation: Proprioception vs Camera")

    # Plot 5: Frame timing consistency
    ax5 = fig.add_subplot(gs[3, :])
    frame_intervals = np.diff(timestamps) * 1000  # Convert to ms
    expected_interval = 1000 / fps

    ax5.plot(timestamps[1:], frame_intervals, "b-", linewidth=1.5, alpha=0.7)
    ax5.axhline(
        y=expected_interval,
        color="g",
        linestyle="--",
        label=f"Expected interval ({expected_interval:.1f}ms @ {fps}Hz)",
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Frame Interval (ms)")
    ax5.set_title(f"Frame Timing Consistency (target: {fps}Hz)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle(
        f"Dataset Alignment Verification: {repo_id} (Episode {episode_idx})",
        fontsize=14,
        fontweight="bold",
    )

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved verification plot to {output_path}")
    else:
        plt.savefig("alignment_verification.png", dpi=150, bbox_inches="tight")
        logger.info("Saved verification plot to alignment_verification.png")

    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify dataset alignment between camera and proprioception",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Full HuggingFace repo_id (e.g., 'giacomoran/so101_data_collection_cube_hand_guided_1')",
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
        "--output",
        type=Path,
        help="Output path for verification plot (default: alignment_verification.png)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    verify_alignment(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        episode_idx=args.episode,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
