#!/usr/bin/env python
"""
Push a locally saved dataset to HuggingFace Hub.

Usage:
    python -m src.push_to_hub --dataset-name cube_hand_guided
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_data_collection.collect.collect import DEFAULT_DATASET_ROOT
from so101_data_collection.shared.setup import HF_REPO_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def push_dataset_to_hub(
    dataset_name: str, dataset_root: Path = DEFAULT_DATASET_ROOT
) -> None:
    """Push a dataset to HuggingFace Hub."""
    dataset_path = dataset_root / dataset_name

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Enforce naming convention: HuggingFace repo names should use underscores, not hyphens
    assert "-" not in HF_REPO_ID, (
        f"HF_REPO_ID '{HF_REPO_ID}' contains hyphens. "
        "Use underscores instead (e.g., 'user_name/repo_name')."
    )
    repo_id = f"{HF_REPO_ID}_{dataset_name}"

    logger.info(f"Loading dataset from: {dataset_path}")
    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")

    # Load existing dataset (repo_id is required even for local datasets)
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)

    # Push to Hub
    try:
        dataset.push_to_hub()
        logger.info("Dataset pushed successfully!")
    except Exception as e:
        logger.error(f"Failed to push dataset to Hub: {e}", exc_info=True)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Push SO-101 dataset to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset directory in /data (e.g., 'cube_hand_guided')",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing datasets",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_dataset_to_hub(args.dataset_name, args.dataset_root)


if __name__ == "__main__":
    main()
