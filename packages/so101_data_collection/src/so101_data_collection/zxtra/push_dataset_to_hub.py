#!/usr/bin/env python
"""
Push a locally saved dataset to HuggingFace Hub.

Usage:
    # Push from HF cache (where preprocess_dataset.py stores output)
    python -m so101_data_collection.zxtra.push_dataset_to_hub \
        --repo-id giacomoran/so101_data_collection_cube_hand_224x11

    # Push from custom directory
    python -m so101_data_collection.zxtra.push_dataset_to_hub \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided \
        --dataset-root /path/to/datasets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def push_dataset_to_hub(repo_id: str, dataset_root: Path = HF_LEROBOT_HOME) -> None:
    """
    Push a dataset to HuggingFace Hub.

    Args:
        repo_id: Full HuggingFace repo_id (e.g., 'giacomoran/so101_data_collection_cube_hand_guided')
        dataset_root: Root directory containing datasets (default: HF_LEROBOT_HOME cache)
    """
    # The dataset is stored at dataset_root / repo_id
    dataset_path = dataset_root / repo_id

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {dataset_path}")
    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")

    # Bypass LeRobotDataset/LeRobotDatasetMetadata constructors because they
    # try to fetch from HuggingFace if local loading fails, and the repo may
    # not exist on HF yet (that's why we're pushing it!)
    try:
        ds_meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
        ds_meta.repo_id = repo_id
        ds_meta.root = dataset_path
        ds_meta.revision = None
        ds_meta.writer = None
        ds_meta.latest_episode = None
        ds_meta.metadata_buffer = []
        ds_meta.metadata_buffer_size = 10
        ds_meta.load_metadata()

        dataset = LeRobotDataset.__new__(LeRobotDataset)
        dataset.repo_id = repo_id
        dataset.root = dataset_path
        dataset.meta = ds_meta
        dataset.revision = None
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        sys.exit(1)

    # Push to Hub
    try:
        dataset.push_to_hub()
        logger.info("Dataset pushed successfully!")
        logger.info(f"View at: https://huggingface.co/datasets/{repo_id}")
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
        default=HF_LEROBOT_HOME,
        help="Root directory containing datasets (default: HF cache where preprocess_dataset.py stores output)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_dataset_to_hub(args.repo_id, args.dataset_root)


if __name__ == "__main__":
    main()
