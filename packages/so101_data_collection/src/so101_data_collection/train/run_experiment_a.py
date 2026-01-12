#!/usr/bin/env python
"""Training script for Experiment A: Prefix Conditioning with Frozen Backbone.

This script runs prefix conditioning experiment, training three variants:
- decoder_pos: Prefix via decoder positional embeddings
- encoder_input: Prefix concatenated to encoder input
- encoder_output: Prefix concatenated to encoder output

Usage:
    python run_experiment_a.py --prefix_mode encoder_input --num_train_steps 2000
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot_policy_act_relative_rtc import ACTRelativeRTCPolicy
from lerobot_policy_act_relative_rtc.processor_act_relative_rtc import (
    ImagePadSquareResizeProcessorStep,
)
from torch.utils.data import DataLoader

from so101_data_collection.train.modeling_act_with_prefix import (
    ACTWithPrefixConfig,
    ACTWithPrefixPolicy,
)
from lerobot_policy_act_relative_rtc import compute_relative_stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add lerobot_policy_act_relative_rtc to path
package_path = (
    Path(__file__).parent.parent.parent.parent
    / "lerobot_policy_act_relative_rtc"
    / "src"
)
sys.path.insert(0, str(package_path))



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A: Prefix Conditioning Probe."""

    prefix_mode: str = "encoder_input"
    pretrained_path: str = (
        "outputs/cube_hand_guided_act_umi_wrist_10_30k/pretrained_model"
    )
    output_dir: str = "outputs/experiment_a"
    max_delay: int = 3
    freeze_backbone: bool = True
    freeze_encoder: bool = True
    freeze_decoder: bool = True
    num_train_steps: int = 2000
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    log_interval: int = 50
    save_interval: int = 500
    seed: int = 42
    downscale_img_square: int | None = None

    @property
    def is_vanilla(self) -> bool:
        """Check if running in vanilla mode (no prefix conditioning)."""
        return self.prefix_mode == "vanilla"


def load_pretrained_policy(
    pretrained_path: str | Path,
    prefix_mode: str,
    device: str = "cpu",
) -> ACTWithPrefixPolicy:
    """Load pretrained ACTRelativeRTC and convert to ACTWithPrefixPolicy.

    Args:
        pretrained_path: Path to pretrained model directory
        prefix_mode: Prefix conditioning mode
        device: Device to load model on

    Returns:
        ACTWithPrefixPolicy instance with pretrained weights loaded
    """
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.is_dir():
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

    logger.info(f"Loading pretrained model from {pretrained_path}")
    pretrained_policy = ACTRelativeRTCPolicy.from_pretrained(
        str(pretrained_path), device=device
    )

    act_config = ACTWithPrefixConfig(
        prefix_mode=prefix_mode,
        **pretrained_policy.config.__dict__,
    )

    policy = ACTWithPrefixPolicy(act_config, dataset_meta=None)

    pretrained_state = pretrained_policy.state_dict()
    current_state = policy.state_dict()

    filtered_pretrained = {}
    for k, v in pretrained_state.items():
        if (
            "prefix_proj" in k
            or "action_prefix_proj" in k
            or "prefix_token_type_embed" in k
        ):
            continue
        if k in current_state:
            filtered_pretrained[k] = v
        elif k.startswith("model.") and k[6:] in current_state:
            filtered_pretrained[k[6:]] = v

    missing_keys, unexpected_keys = policy.load_state_dict(
        filtered_pretrained, strict=False
    )
    if missing_keys:
        logger.info(f"Missing keys (expected for new prefix modules): {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    policy = policy.to(device)
    logger.info("Pretrained weights loaded successfully")

    return policy


def make_preprocessor(
    config,
    policy,
    dataset_stats: dict[str, Any] | None = None,
    device: str = "cpu",
):
    """Create the preprocessing pipeline for training.

    Uses the same processors as the original ACT Relative RTC model:
    1. Rename observations (if needed)
    2. Add batch dimension
    3. Move to device
    4. Pad and resize images
    5. Normalize observations and actions

    Args:
        config: Experiment configuration
        policy: ACTWithPrefixPolicy with config
        dataset_stats: Optional dataset statistics for normalization
        device: Device to move data to

    Returns:
        Preprocessor function that transforms batches (dict format for policy.forward)
    """
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=device),
        ImagePadSquareResizeProcessorStep(
            downscale_img_square=config.downscale_img_square
        ),
        NormalizerProcessorStep(
            features={**policy.config.input_features, **policy.config.output_features},
            norm_map=policy.config.normalization_mapping,
            stats=dataset_stats,
            device=device,
        ),
    ]

    pipeline = PolicyProcessorPipeline(
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    def preprocessor_fn(batch):
        return pipeline(batch)

    return preprocessor_fn


def freeze_model(
    policy: ACTWithPrefixPolicy,
    freeze_backbone: bool = True,
    freeze_encoder: bool = True,
    freeze_decoder: bool = True,
):
    """Freeze parameters in model.

    Only action_prefix_proj remains trainable. Normalizers have no parameters (just buffers).

    Args:
        policy: ACTWithPrefixPolicy to freeze
        freeze_backbone: Freeze image backbone
        freeze_encoder: Freeze transformer encoder
        freeze_decoder: Freeze transformer decoder
    """
    for param in policy.parameters():
        param.requires_grad = False

    for param in policy.model.action_prefix_proj.parameters():
        param.requires_grad = True

    if freeze_backbone and hasattr(policy.model, "backbone"):
        for param in policy.model.backbone.parameters():
            param.requires_grad = False

    if freeze_encoder:
        for param in policy.model.encoder.parameters():
            param.requires_grad = False

    if freeze_decoder:
        for param in policy.model.decoder.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )


def train_prefix_conditioning(
    config: ExperimentAConfig,
    dataset,
    repo_id: str,
):
    """Train prefix conditioning model.

    Args:
        config: Experiment configuration
        dataset: Training dataset (LeRobotDataset)
        repo_id: Dataset repository ID

    Returns:
        metrics: Training metrics dictionary
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    policy = load_pretrained_policy(config.pretrained_path, config.prefix_mode, device)

    fps = dataset.fps
    chunk_size = policy.config.chunk_size
    obs_state_delta_frames = policy.config.obs_state_delta_frames

    delta_timestamps = {
        "observation.state": [-obs_state_delta_frames / fps, 0],
        "action": [i / fps for i in range(chunk_size)],
    }

    if policy.config.image_features:
        for key in policy.config.image_features:
            delta_timestamps[key] = [-obs_state_delta_frames / fps, 0]

    root = getattr(dataset, "root", None)
    stats_dataset = LeRobotDataset(
        repo_id, root=root, delta_timestamps=delta_timestamps
    )

    if not policy.has_relative_stats:
        logger.info("Computing relative stats from training dataset...")
        stats = compute_relative_stats(
            stats_dataset,
            batch_size=64,
            num_workers=4,
        )
        policy.set_relative_stats(stats)
        logger.info("Relative stats configured successfully.")

    freeze_model(
        policy, config.freeze_backbone, config.freeze_encoder, config.freeze_decoder
    )
    policy.train()

    logger.info("Creating preprocessing pipeline...")
    preprocessor = make_preprocessor(
        config=config,
        policy=policy,
        dataset_stats=None,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    root = getattr(dataset, "root", None)
    training_dataset = LeRobotDataset(
        repo_id, root=root, delta_timestamps=delta_timestamps
    )

    dataloader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device != "cpu",
        drop_last=True,
    )

    metrics = {
        "train_loss": [],
        "train_postfix_error": [],
        "train_prefix_error": [],
        "train_boundary_diff": [],
        "step": [],
    }

    output_dir = Path(config.output_dir) / config.prefix_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Prefix mode: {config.prefix_mode}")
    logger.info(f"Max delay: {config.max_delay}")
    logger.info(f"Training steps: {config.num_train_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Output directory: {output_dir}")
    if config.is_vanilla:
        logger.info("Vanilla mode: Using delay=0 (no prefix conditioning)")
    logger.info("=" * 60)

    step = 0
    epoch = 0

    while step < config.num_train_steps:
        epoch += 1
        logger.info(f"Epoch {epoch}")

        for batch_idx, batch in enumerate(dataloader):
            if step >= config.num_train_steps:
                break

            if config.is_vanilla:
                delay = 0
            else:
                delay = random.randint(1, config.max_delay)

            batch = preprocessor(batch)

            loss, loss_dict = policy.forward(batch, delay=delay)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics["train_loss"].append(loss.item())
            metrics["train_postfix_error"].append(loss_dict.get("l1_loss_postfix", 0.0))
            metrics["train_prefix_error"].append(loss_dict.get("l1_loss_prefix", 0.0))
            metrics["train_boundary_diff"].append(loss_dict.get("boundary_diff", 0.0))
            metrics["step"].append(step)

            if step % config.log_interval == 0:
                logger.info(
                    f"Step {step}/{config.num_train_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"L1 Loss Original: {loss_dict.get('l1_loss_original', 0.0):.4f} | "
                    f"Postfix Error: {loss_dict.get('l1_loss_postfix', 0.0):.4f} | "
                    f"Prefix Error: {loss_dict.get('l1_loss_prefix', 0.0):.4f} | "
                    f"Boundary Diff: {loss_dict.get('boundary_diff', 0.0):.4f} | "
                    f"Delay: {delay}"
                )

            if step > 0 and step % config.save_interval == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            step += 1

    final_model_path = output_dir / "final_policy.pt"
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        final_model_path,
    )
    logger.info(f"Saved final model to {final_model_path}")

    metrics_path = output_dir / "metrics.json"
    metrics_json = {
        k: [float(x) for x in v] if isinstance(v, list) else v
        for k, v in metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    logger.info(f"Final postfix error: {metrics['train_postfix_error'][-1]:.4f}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ACT with prefix conditioning (Experiment A)"
    )
    parser.add_argument(
        "--prefix_mode",
        type=str,
        default="encoder_input",
        choices=["vanilla", "decoder_pos", "encoder_input", "encoder_output"],
        help="Prefix conditioning strategy (vanilla uses delay=0 without prefix)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="outputs/cube_hand_guided_act_umi_wrist_10_30k/pretrained_model",
        help="Path to pretrained ACT model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/experiment_a",
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--max_delay",
        type=int,
        default=3,
        help="Maximum delay (sampled uniformly from [1, max_delay])",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze image encoders and transformer encoder",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze transformer encoder",
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        default=True,
        help="Freeze transformer decoder",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=2000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for unfrozen parameters",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=2000,
        help="Checkpoint saving interval (steps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--downscale_img_square",
        type=int,
        default=224,
        help="Target square resolution for image preprocessing (e.g., 224 for 224x224). If None, no resizing.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="giacomoran/so101_data_collection_cube_hand_guided",
        help="Dataset repo ID for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (cuda or cpu)",
    )
    return parser.parse_args()


def load_dataset(repo_id: str, split: str = "train"):
    """Load training dataset.

    Args:
        repo_id: Hugging Face dataset repo ID
        split: Dataset split (train or eval) - currently unused, loads all episodes

    Returns:
        LeRobotDataset instance
    """
    logger.info(f"Loading dataset from {repo_id} (split: {split})")

    dataset = LeRobotDataset(repo_id)

    logger.info(f"Dataset loaded: {len(dataset)} frames")
    logger.info(f"FPS: {dataset.fps}")

    if hasattr(dataset, "num_episodes"):
        logger.info(f"Number of episodes: {dataset.num_episodes}")

    try:
        if hasattr(dataset, "features"):
            features = dataset.features.keys()
            image_keys = [k for k in features if "image" in k.lower()]
            state_keys = [k for k in features if "state" in k.lower()]
            action_keys = [k for k in features if "action" in k.lower()]
            if image_keys:
                logger.info(f"Image keys: {image_keys}")
            if state_keys:
                logger.info(f"State keys: {state_keys}")
            if action_keys:
                logger.info(f"Action keys: {action_keys}")
    except Exception:
        pass

    return dataset


def verify_setup(config: ExperimentAConfig, device: str):
    """Verify that model setup is correct before training.

    Args:
        config: Experiment configuration
        device: Device to use

    Returns:
        bool: True if verification passed
    """
    logger.info("=" * 60)
    logger.info("VERIFYING SETUP")
    logger.info("=" * 60)

    pretrained_path = Path(config.pretrained_path)
    if not pretrained_path.is_dir():
        logger.error(f"Pretrained model not found at {config.pretrained_path}")
        return False

    config_file = pretrained_path / "config.json"
    if not config_file.exists():
        logger.error(f"Config file not found at {config_file}")
        return False

    logger.info(f"✓ Pretrained model found at {config.pretrained_path}")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    logger.info(f"✓ Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir}")

    logger.info("=" * 60)
    logger.info("VERIFICATION PASSED")
    logger.info("=" * 60)

    return True


def main():
    args = parse_args()

    config = ExperimentAConfig(
        prefix_mode=args.prefix_mode,
        pretrained_path=args.pretrained_path,
        output_dir=args.output_dir,
        max_delay=args.max_delay,
        freeze_backbone=args.freeze_backbone,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        downscale_img_square=args.downscale_img_square,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT A: PREFIX CONDITIONING PROBE")
    logger.info("=" * 60)
    logger.info(f"Prefix mode: {config.prefix_mode}")
    logger.info(f"Max delay: {config.max_delay}")
    logger.info(f"Training steps: {config.num_train_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Freeze backbone: {config.freeze_backbone}")
    logger.info(f"Freeze encoder: {config.freeze_encoder}")
    logger.info(f"Freeze decoder: {config.freeze_decoder}")
    logger.info(f"Seed: {config.seed}")
    logger.info("=" * 60)

    if not verify_setup(config, args.device):
        logger.error("Setup verification failed. Exiting.")
        sys.exit(1)

    train_dataset = load_dataset(args.dataset_repo_id, split="train")

    train_prefix_conditioning(config, train_dataset, args.dataset_repo_id)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(
        f"Final metrics saved to {config.output_dir}/{config.prefix_mode}/metrics.json"
    )
    logger.info(
        f"Final model saved to {config.output_dir}/{config.prefix_mode}/final_policy.pt"
    )


if __name__ == "__main__":
    main()
