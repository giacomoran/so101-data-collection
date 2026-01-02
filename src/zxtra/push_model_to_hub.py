#!/usr/bin/env python
"""
Manually push a trained ACT UMI model to HuggingFace Hub.

Usage:
    python src/push_model_to_hub.py \
        --checkpoint_dir=outputs/train_act_umi_overfit/checkpoints/last/pretrained_model \
        --repo_id=giacomoran/my_model_name \
        --commit_message="Training completed"
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from lerobot.configs.types import FeatureType, PolicyFeature
from safetensors.torch import load_file

from act_umi import ACTUMIConfig, ACTUMIPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _convert_features_dict(config_dict: dict) -> dict:
    """Convert feature dictionaries from JSON to PolicyFeature objects."""
    for features_key in ["input_features", "output_features"]:
        if features_key in config_dict:
            features = {}
            for key, ft_dict in config_dict[features_key].items():
                feature_type_map = {
                    "VISUAL": FeatureType.VISUAL,
                    "STATE": FeatureType.STATE,
                    "ACTION": FeatureType.ACTION,
                    "ENV": FeatureType.ENV,
                }
                features[key] = PolicyFeature(
                    type=feature_type_map[ft_dict["type"]],
                    shape=tuple(ft_dict["shape"]),
                )
            config_dict[features_key] = features
    return config_dict


def push_model_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    commit_message: str = "Manual push",
    device: str | None = None,
):
    """Load a model from checkpoint and push it to HuggingFace Hub."""

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    config_path = checkpoint_path / "config.json"
    model_path = checkpoint_path / SAFETENSORS_SINGLE_FILE

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Auto-select device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Loading model from {checkpoint_path}")
    logger.info(f"Using device: {device}")

    # Load config
    import json

    with open(config_path) as f:
        config_dict = json.load(f)

    # Remove type field if present (used by draccus for polymorphism)
    config_dict.pop("type", None)

    # Convert feature dictionaries to PolicyFeature objects
    config_dict = _convert_features_dict(config_dict)

    # Create config from dict
    config = ACTUMIConfig(**config_dict)
    config.device = device

    # Create policy
    policy = ACTUMIPolicy(config)

    # Load weights
    # Use strict=False to allow loading buffers (delta_obs_mean, etc.) that might be None initially
    state_dict = load_file(str(model_path))
    policy.load_state_dict(state_dict, strict=False)
    policy = policy.to(device)
    policy.eval()

    logger.info(f"Loaded policy from {checkpoint_path}")
    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")

    # Push to Hub
    try:
        policy.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Successfully pushed model to {repo_id}")
    except Exception as e:
        logger.error(f"Failed to push model to Hub: {e}", exc_info=True)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Push a trained ACT UMI model to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing config.json and model.safetensors",
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
        "--device",
        type=str,
        default=None,
        help="Device to load model on (cuda/mps/cpu). Auto-detected if not specified.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_model_to_hub(
        checkpoint_dir=Path(args.checkpoint_dir),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        device=args.device,
    )


if __name__ == "__main__":
    main()
