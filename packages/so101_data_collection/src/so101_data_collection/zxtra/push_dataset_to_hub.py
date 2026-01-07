#!/usr/bin/env python
"""
Push a locally saved dataset to HuggingFace Hub.

Usage:
    python -m src.push_to_hub --repo-id giacomoran/so101_data_collection_cube_hand_guided
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_data_collection.collect.collect import DEFAULT_DATASET_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def push_dataset_to_hub(
    repo_id: str, dataset_root: Path = DEFAULT_DATASET_ROOT
) -> None:
    """
    Push a dataset to HuggingFace Hub.

    Args:
        repo_id: Full HuggingFace repo_id (e.g., 'giacomoran/so101_data_collection_cube_hand_guided')
        dataset_root: Root directory containing datasets
    """
    # The dataset is stored at dataset_root / repo_id
    dataset_path = dataset_root / repo_id

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

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
        "--repo-id",
        type=str,
        required=True,
        help="Full HuggingFace repo_id (e.g., 'giacomoran/so101_data_collection_cube_hand_guided')",
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
    push_dataset_to_hub(args.repo_id, args.dataset_root)


if __name__ == "__main__":
    main()
