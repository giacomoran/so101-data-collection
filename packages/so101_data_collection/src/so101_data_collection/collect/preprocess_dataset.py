#!/usr/bin/env python
"""Preprocess LeRobot datasets for ACT Relative RTC training.

This tool prepares datasets for training by:
1. Precomputing relative stats (delta_obs, relative_action) and storing in meta/relative_stats.json
2. Preprocessing images to target resolution (pad to square + resize) using ffmpeg

This eliminates runtime overhead from:
- ~2min relative stats computation at every training init
- GPU utilization issues from runtime image resizing

Usage:
    # Full preprocessing (create new dataset with resized videos + relative stats)
    python -m so101_data_collection.collect.preprocess_dataset \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided \
        --output-repo-id giacomoran/cube_hand_guided_224 \
        --target-resolution 224

    # Relative stats only (in-place, adds meta/relative_stats.json)
    python -m so101_data_collection.collect.preprocess_dataset \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided \
        --relative-stats-only

    # Push to HuggingFace Hub
    python -m so101_data_collection.collect.preprocess_dataset \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided \
        --output-repo-id giacomoran/cube_hand_guided_224 \
        --target-resolution 224 \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for dataset preprocessing."""

    repo_id: str
    output_repo_id: str | None = None
    target_resolution: int | None = None
    relative_stats_only: bool = False
    push_to_hub: bool = False
    skip_confirmation: bool = False
    obs_state_delta_frames: int = 1
    chunk_size: int = 100
    batch_size: int = 64
    num_workers: int = 4


def compute_and_save_relative_stats(
    ds_meta: LeRobotDatasetMetadata,
    output_path: Path,
    config: PreprocessConfig,
) -> dict:
    """Compute relative stats and save to meta/relative_stats.json.

    This reuses the logic from lerobot_policy_act_relative_rtc.relative_stats
    but saves to a JSON file.

    Args:
        ds_meta: Dataset metadata.
        output_path: Path to the dataset root (where meta/ folder is).
        config: Preprocessing configuration.

    Returns:
        The computed stats dictionary.
    """
    from lerobot_policy_act_relative_rtc.relative_stats import compute_relative_stats

    logger.info("Computing relative stats...")

    # Build delta_timestamps for relative stats computation
    fps = ds_meta.fps
    delta_indices_state = [-config.obs_state_delta_frames, 0]
    delta_indices_action = list(range(config.chunk_size))

    delta_timestamps = {}
    for key in ds_meta.features:
        if key == ACTION:
            delta_timestamps[key] = [i / fps for i in delta_indices_action]
        elif key == OBS_STATE:
            delta_timestamps[key] = [i / fps for i in delta_indices_state]
        elif key.startswith("observation."):
            # For images and other observations, use state_delta_indices
            delta_timestamps[key] = [i / fps for i in delta_indices_state]

    # Create dataset with appropriate delta_timestamps
    dataset = LeRobotDataset(
        ds_meta.repo_id,
        root=ds_meta.root,
        delta_timestamps=delta_timestamps,
    )

    # Compute relative stats
    stats = compute_relative_stats(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Convert numpy arrays to lists for JSON serialization
    stats_json = {
        "delta_obs": {
            "mean": stats["delta_obs"]["mean"].tolist(),
            "std": stats["delta_obs"]["std"].tolist(),
        },
        "relative_action": {
            "mean": stats["relative_action"]["mean"].tolist(),
            "std": stats["relative_action"]["std"].tolist(),
        },
        "config": {
            "obs_state_delta_frames": config.obs_state_delta_frames,
            "chunk_size": config.chunk_size,
            "total_samples": len(dataset),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Save to meta/relative_stats.json
    meta_path = output_path / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)
    stats_file = meta_path / "relative_stats.json"

    with open(stats_file, "w") as f:
        json.dump(stats_json, f, indent=2)

    logger.info(f"Saved relative stats to {stats_file}")
    return stats_json


def resize_video_ffmpeg(
    path_input: Path,
    path_output: Path,
    target_resolution: int,
) -> None:
    """Resize a video using ffmpeg with pad-to-square then scale.

    Uses ffmpeg's pad and scale filters directly on video files, which is
    much faster than decoding frames, processing in Python, and re-encoding.

    Args:
        path_input: Path to input video file.
        path_output: Path to output video file.
        target_resolution: Target square resolution (e.g., 224 for 224x224).
    """
    # ffmpeg filter: pad to square, then scale
    # pad=max(iw,ih):max(iw,ih):(ow-iw)/2:(oh-ih)/2:black pads to square
    # scale=target:target resizes to target resolution
    filter_str = (
        f"pad=max(iw\\,ih):max(iw\\,ih):(ow-iw)/2:(oh-ih)/2:black,scale={target_resolution}:{target_resolution}"
    )

    # Ensure output directory exists
    path_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i",
        str(path_input),
        "-vf",
        filter_str,
        "-c:v",
        "libx264",  # H.264 codec
        "-preset",
        "fast",  # Balance speed/quality
        "-crf",
        "23",  # Quality (lower = better, 23 is default)
        "-an",  # No audio
        str(path_output),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed to process {path_input}")


def copy_and_resize_dataset(
    ds_meta: LeRobotDatasetMetadata,
    path_output: Path,
    target_resolution: int,
) -> None:
    """Copy dataset to new location and resize all videos.

    Args:
        ds_meta: Source dataset metadata.
        path_output: Output path for the new dataset.
        target_resolution: Target square resolution for videos.
    """
    path_source = Path(ds_meta.root)

    logger.info(f"Creating preprocessed dataset at {path_output}")

    # Create output directory
    path_output.mkdir(parents=True, exist_ok=True)

    # Copy non-video files (meta/, data/)
    for item in path_source.iterdir():
        path_dest = path_output / item.name
        if item.name == "videos":
            # Handle videos separately
            continue
        elif item.is_dir():
            if path_dest.exists():
                shutil.rmtree(path_dest)
            shutil.copytree(item, path_dest)
            logger.info(f"Copied {item.name}/")
        else:
            shutil.copy2(item, path_dest)
            logger.info(f"Copied {item.name}")

    # Process videos
    path_videos_source = path_source / "videos"
    path_videos_output = path_output / "videos"

    if path_videos_source.exists():
        # Find all video files
        files_video = list(path_videos_source.rglob("*.mp4"))
        logger.info(f"Found {len(files_video)} video files to process")

        for path_video in tqdm(files_video, desc="Resizing videos"):
            # Maintain relative path structure
            path_relative = path_video.relative_to(path_videos_source)
            path_video_output = path_videos_output / path_relative

            resize_video_ffmpeg(path_video, path_video_output, target_resolution)

    # Update info.json to reflect new repo_id if needed
    path_info = path_output / "meta" / "info.json"
    if path_info.exists():
        with open(path_info) as f:
            info = json.load(f)

        # Update image feature shapes in info.json
        # LeRobot uses [height, width, channels] format for video features
        if "features" in info:
            for key, feature in info["features"].items():
                if "image" in key and "shape" in feature:
                    # Original shape is [H, W, C], update H and W to target_resolution
                    channels = feature["shape"][2]  # C is the last element
                    feature["shape"] = [target_resolution, target_resolution, channels]
                    # Also update video.height and video.width in info if present
                    if "info" in feature:
                        feature["info"]["video.height"] = target_resolution
                        feature["info"]["video.width"] = target_resolution
                    logger.info(f"Updated {key} shape to {feature['shape']}")

        with open(path_info, "w") as f:
            json.dump(info, f, indent=2)
        logger.info("Updated meta/info.json with new image shapes")


def preprocess_dataset(config: PreprocessConfig) -> None:
    """Main preprocessing orchestration function.

    Args:
        config: Preprocessing configuration.
    """
    logger.info("=" * 80)
    logger.info("Dataset Preprocessing for ACT Relative RTC")
    logger.info("=" * 80)
    logger.info(f"Source dataset: {config.repo_id}")

    if config.relative_stats_only:
        logger.info("Mode: relative stats only (in-place)")
    else:
        logger.info(f"Output dataset: {config.output_repo_id}")
        logger.info(f"Target resolution: {config.target_resolution}")

    logger.info("")

    # Load source dataset metadata
    ds_meta = LeRobotDatasetMetadata(config.repo_id)
    logger.info(f"Dataset FPS: {ds_meta.fps}")
    logger.info(f"Total episodes: {ds_meta.total_episodes}")
    logger.info(f"Total frames: {ds_meta.total_frames}")
    logger.info(f"Dataset root: {ds_meta.root}")
    logger.info("")

    # Show critical parameters that affect relative stats and ask for confirmation
    logger.info("=" * 80)
    logger.info("CRITICAL: Relative stats parameters (must match training config!)")
    logger.info("=" * 80)
    logger.info(f"  chunk_size:             {config.chunk_size}")
    logger.info(f"  obs_state_delta_frames: {config.obs_state_delta_frames}")
    logger.info("")
    logger.info("These values MUST match your training configuration.")
    logger.info("Different values will produce incompatible relative stats.")
    logger.info("=" * 80)

    if not config.skip_confirmation:
        response = input("\nProceed with these parameters? [y/N]: ").strip().lower()
        if response != "y":
            logger.info("Aborted by user.")
            return
        logger.info("")

    if config.relative_stats_only:
        # Just compute and save relative stats in-place
        compute_and_save_relative_stats(
            ds_meta,
            Path(ds_meta.root),
            config,
        )
    else:
        # Full preprocessing: copy dataset, resize videos, compute stats
        if config.output_repo_id is None:
            raise ValueError("--output-repo-id is required for full preprocessing")
        if config.target_resolution is None:
            raise ValueError("--target-resolution is required for full preprocessing")

        # Determine output path
        # Use same parent as source, with new repo_id
        path_source = Path(ds_meta.root)
        path_output = path_source.parent.parent / config.output_repo_id

        # Copy and resize dataset
        copy_and_resize_dataset(ds_meta, path_output, config.target_resolution)

        # Compute and save relative stats in the new dataset
        ds_meta_output = LeRobotDatasetMetadata(
            config.output_repo_id,
            root=path_output,
        )
        compute_and_save_relative_stats(
            ds_meta_output,
            path_output,
            config,
        )

        logger.info("")
        logger.info(f"Preprocessed dataset created at: {path_output}")

        # Push to hub if requested
        if config.push_to_hub:
            logger.info("")
            logger.info("Pushing to HuggingFace Hub...")
            dataset = LeRobotDataset(
                config.output_repo_id,
                root=path_output,
            )
            dataset.push_to_hub()
            logger.info(f"Pushed to hub: {config.output_repo_id}")

    logger.info("")
    logger.info("Preprocessing complete!")


def parse_args() -> PreprocessConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess LeRobot dataset for ACT Relative RTC training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Source dataset repository ID (e.g., 'giacomoran/cube_hand_guided')",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output dataset repository ID (required unless --relative-stats-only)",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=None,
        help="Target square resolution for images (e.g., 224)",
    )
    parser.add_argument(
        "--relative-stats-only",
        action="store_true",
        help="Only compute relative stats (in-place, no video resizing)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push preprocessed dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="skip_confirmation",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--obs-state-delta-frames",
        type=int,
        default=1,
        help="Number of frames for observation delta computation",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Action chunk size for relative stats computation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for stats computation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )

    args = parser.parse_args()

    # Validation
    if not args.relative_stats_only:
        if args.output_repo_id is None:
            parser.error("--output-repo-id is required unless --relative-stats-only")
        if args.target_resolution is None:
            parser.error("--target-resolution is required unless --relative-stats-only")

    return PreprocessConfig(
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        target_resolution=args.target_resolution,
        relative_stats_only=args.relative_stats_only,
        push_to_hub=args.push_to_hub,
        skip_confirmation=args.skip_confirmation,
        obs_state_delta_frames=args.obs_state_delta_frames,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def main():
    init_logging()
    config = parse_args()
    preprocess_dataset(config)


if __name__ == "__main__":
    main()
