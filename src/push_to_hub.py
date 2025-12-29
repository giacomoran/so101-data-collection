#!/usr/bin/env python
"""
Push a locally saved dataset to HuggingFace Hub.

Usage:
    python -m src.push_to_hub \
        --task pick_place_cube \
        --method hand_guided
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from src.collect import DEFAULT_DATASET_ROOT, Method, Task  # noqa: E402
from src.setup import HF_REPO_ID  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def push_dataset_to_hub(
    task: Task, method: Method, dataset_root: Path = DEFAULT_DATASET_ROOT
) -> None:
    """Push a dataset to HuggingFace Hub."""
    # Construct dataset name and paths (same as collect.py)
    dataset_name = f"{task.value}_{method.value}"
    repo_id = f"{HF_REPO_ID}-{dataset_name}"
    dataset_path = dataset_root / dataset_name

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
        "--task",
        type=str,
        required=True,
        choices=[t.value for t in Task],
        help="Task name",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[m.value for m in Method],
        help="Collection method",
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
    task = Task(args.task)
    method = Method(args.method)
    push_dataset_to_hub(task, method, args.dataset_root)


if __name__ == "__main__":
    main()
