#!/usr/bin/env python
"""
Manually push a trained ACT UMI model to HuggingFace Hub.

This script pushes the complete model checkpoint including:
- Model weights (model.safetensors)
- Policy config (config.json)
- Preprocessor pipeline with normalization stats (policy_preprocessor.json + .safetensors)
- Postprocessor pipeline with unnormalization stats (policy_postprocessor.json + .safetensors)

Usage:
    python src/zxtra/push_model_to_hub.py \
        --checkpoint_dir=outputs/train_act_umi_overfit/checkpoints/last/pretrained_model \
        --repo_id=giacomoran/my_model_name \
        --commit_message="Training completed"
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def push_model_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    commit_message: str = "Manual push",
    private: bool = False,
):
    """Push a complete model checkpoint to HuggingFace Hub.

    This uploads all files from the checkpoint directory including:
    - Model weights (model.safetensors)
    - Policy config (config.json)
    - Preprocessor pipeline (policy_preprocessor.json + normalization stats)
    - Postprocessor pipeline (policy_postprocessor.json + unnormalization stats)

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/last/pretrained_model)
        repo_id: HuggingFace Hub repository ID (e.g., 'username/model_name')
        commit_message: Commit message for the push
        private: Whether to create a private repository
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Verify required files exist
    config_path = checkpoint_path / "config.json"
    model_path = checkpoint_path / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Check for processor files (warn if missing)
    preprocessor_path = checkpoint_path / "policy_preprocessor.json"
    postprocessor_path = checkpoint_path / "policy_postprocessor.json"

    if not preprocessor_path.exists():
        logger.warning(
            f"Preprocessor config not found at {preprocessor_path}. "
            "Model may be missing normalization stats!"
        )
    if not postprocessor_path.exists():
        logger.warning(
            f"Postprocessor config not found at {postprocessor_path}. "
            "Model may be missing unnormalization stats!"
        )

    # List files to be uploaded
    files_to_upload = list(checkpoint_path.glob("*.json")) + list(
        checkpoint_path.glob("*.safetensors")
    )
    # Also include README if present
    readme_path = checkpoint_path / "README.md"
    if readme_path.exists():
        files_to_upload.append(readme_path)

    logger.info(f"Found {len(files_to_upload)} files to upload:")
    for f in sorted(files_to_upload):
        logger.info(f"  - {f.name}")

    # Create/get repository and upload
    api = HfApi()

    logger.info(f"Creating/accessing repository: {repo_id}")
    repo_url = api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    logger.info(f"Repository URL: {repo_url}")

    logger.info(f"Uploading files from {checkpoint_path}")
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint_path),
        commit_message=commit_message,
        allow_patterns=["*.safetensors", "*.json", "*.md"],
        ignore_patterns=["*.tmp", "*.log"],
    )

    logger.info(f"Successfully pushed model to {commit_info.repo_url}")
    logger.info(f"Commit: {commit_info.commit_url}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Push a trained ACT UMI model to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing config.json, model.safetensors, and processor files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., 'username/model_name')",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Manual push",
        help="Commit message for the push",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_model_to_hub(
        checkpoint_dir=Path(args.checkpoint_dir),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        private=args.private,
    )


if __name__ == "__main__":
    main()
