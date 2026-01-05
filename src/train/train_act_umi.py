#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modified 2025 by Giacomo Randazzo for ACT with relative joint positions (UMI-style).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for ACT with Relative Joint Positions (UMI-style).

This script trains the ACTUMIPolicy on a LeRobot dataset.

Key differences from standard ACT training:
- Uses delta_timestamps to fetch obs.state[t-1] and obs.state[t]
- The policy internally converts absolute actions to relative actions
- The policy uses observation deltas instead of absolute observations

Usage:
    python src/train_act_umi.py \
        --dataset.repo_id=giacomoran/cube_hand_guided \
        --policy.chunk_size=50 \
        --steps=50000 \
        --batch_size=8 \
        --wandb.enable=true

Training with wrist camera only:
    python src/train_act_umi.py \
        --dataset.repo_id=giacomoran/cube_hand_guided \
        --policy.input_features='{"observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [6]}}'
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import cycle
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OBS_PREFIX,
    OBS_STATE,
)
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging
from safetensors.torch import load_file
from torch.optim import Optimizer

# Import our ACT UMI policy
from act_umi import (
    ACTUMIConfig,
    ACTUMIPolicy,
    compute_relative_stats,
    make_act_umi_pre_post_processors,
)


class WandBLogger:
    """A helper class to log training metrics to wandb."""

    def __init__(self, cfg: "TrainACTUMIConfig"):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        self._wandb = wandb

        # Build tags
        tags = [
            "policy:act_umi",
            f"seed:{cfg.seed}",
            f"dataset:{cfg.dataset.repo_id}",
        ]

        wandb.init(
            id=self.cfg.run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=tags,
            dir=self.log_dir,
            config=draccus.encode(cfg),
            save_code=False,
            job_type="train",
            mode=self.cfg.mode
            if self.cfg.mode in ["online", "offline", "disabled"]
            else "online",
        )

        # Store run_id for resumption
        cfg.wandb.run_id = wandb.run.id

        logging.info(f"WandB initialized. Track this run --> {wandb.run.get_url()}")

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        """Log a dictionary of metrics."""
        for k, v in d.items():
            if isinstance(v, (int, float, str)):
                self._wandb.log({f"{mode}/{k}": v}, step=step)

    def log_policy(self, checkpoint_dir: Path):
        """Log policy checkpoint as artifact."""
        if self.cfg.disable_artifact:
            return
        step_id = checkpoint_dir.name
        artifact_name = f"act_umi-{self.job_name or 'model'}-{step_id}".replace(
            "/", "_"
        ).replace(":", "_")
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_dir = checkpoint_dir / "pretrained_model"
        if pretrained_dir.exists():
            artifact.add_dir(str(pretrained_dir))
            self._wandb.log_artifact(artifact)


@dataclass
class TrainACTUMIConfig:
    """Configuration for training ACT with Relative Joint Positions (UMI-style)."""

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Policy configuration (will be populated from dataset)
    policy: ACTUMIConfig = field(default_factory=ACTUMIConfig)

    # WandB configuration
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # Training configuration
    steps: int = 50000
    batch_size: int = 8
    num_workers: int = 4
    seed: int | None = 42

    # Optimizer configuration
    grad_clip_norm: float = 10.0

    # Logging and checkpointing
    output_dir: Path = Path("outputs/train_act_umi")
    job_name: str | None = None
    log_freq: int = 100
    save_freq: int = 5000

    # Tolerance for timestamp matching
    tolerance_s: float = 1e-4

    # Device
    device: str | None = None

    # Resume from checkpoint
    resume: bool = False


def get_last_checkpoint_dir(output_dir: Path) -> Path | None:
    """Get the last checkpoint directory if it exists."""
    last_checkpoint = output_dir / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    if last_checkpoint.exists():
        # Resolve the symlink
        return last_checkpoint.resolve()
    return None


def load_policy_weights_from_checkpoint(
    policy: ACTUMIPolicy, checkpoint_dir: Path
) -> None:
    """Load policy weights from a checkpoint directory."""
    from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    model_path = checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    state_dict = load_file(str(model_path))

    # Check for key mismatches before loading
    model_keys = set(policy.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    missing_in_checkpoint = model_keys - checkpoint_keys
    unexpected_in_checkpoint = checkpoint_keys - model_keys

    if missing_in_checkpoint:
        logging.warning(
            f"Keys in model but NOT in checkpoint ({len(missing_in_checkpoint)}): {sorted(missing_in_checkpoint)}"
        )
    if unexpected_in_checkpoint:
        logging.warning(
            f"Keys in checkpoint but NOT in model ({len(unexpected_in_checkpoint)}): {sorted(unexpected_in_checkpoint)}"
        )

    matched_keys = model_keys & checkpoint_keys
    logging.info(
        f"Loading {len(matched_keys)}/{len(model_keys)} model keys from checkpoint"
    )

    if len(matched_keys) == 0:
        raise ValueError(
            "No matching keys between model and checkpoint! Weights won't load."
        )

    # Load with strict=False but we've already warned about mismatches
    policy.load_state_dict(state_dict, strict=False)
    logging.info(f"Loaded policy weights from {model_path}")


def resolve_delta_timestamps_for_umi(
    config: ACTUMIConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list[float]]:
    """Resolves delta_timestamps for ACTUMIPolicy.

    Returns delta_timestamps that include:
    - observation.state: Uses state_delta_indices (e.g., [-3/fps, 0] for obs_state_delta_frames=3)
    - observation.images.*: Uses image_delta_indices ([0]) - only current frame needed
    - action: [0, 1/fps, 2/fps, ...] for the action chunk

    This optimization avoids loading intermediate image frames that aren't used,
    while still allowing larger state deltas for better velocity estimation.
    """
    delta_timestamps = {}

    for key in ds_meta.features:
        if key == ACTION and config.action_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in config.action_delta_indices
            ]
        elif key == OBS_STATE:
            # State needs two timestamps for delta computation
            delta_timestamps[key] = [
                i / ds_meta.fps for i in config.state_delta_indices
            ]
        elif key.startswith("observation.images."):
            # Images only need current frame (no delta computation)
            delta_timestamps[key] = [
                i / ds_meta.fps for i in config.image_delta_indices
            ]
        elif key.startswith(OBS_PREFIX):
            # Other observation features (e.g., environment_state) - use image indices
            delta_timestamps[key] = [
                i / ds_meta.fps for i in config.image_delta_indices
            ]

    return delta_timestamps


def make_dataset_for_umi_policy(
    cfg: TrainACTUMIConfig,
) -> LeRobotDataset:
    """Creates a dataset configured for ACTUMIPolicy training.

    Key difference: Uses delta_timestamps to fetch obs.state[t-1] and obs.state[t].
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )

    # Get delta_timestamps for UMI policy (includes obs.state[t-1])
    delta_timestamps = resolve_delta_timestamps_for_umi(cfg.policy, ds_meta)

    logging.info(f"Using delta_timestamps: {delta_timestamps}")

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    return dataset


def make_policy_from_dataset(
    cfg: TrainACTUMIConfig,
    ds_meta: LeRobotDatasetMetadata,
) -> ACTUMIPolicy:
    """Creates an ACTUMIPolicy configured from dataset metadata.

    If cfg.policy.input_features is already set (e.g., via CLI), it will be used.
    Otherwise, input features are inferred from the dataset metadata.
    This allows training with a subset of cameras (e.g., wrist camera only).
    """

    # Check if input_features was already provided via CLI
    if cfg.policy.input_features:
        input_features = cfg.policy.input_features
        logging.info(
            "Using user-provided input_features (e.g., for single-camera training)"
        )
    else:
        # Build input features from dataset
        input_features = {}
        for key, ft_meta in ds_meta.features.items():
            if key == OBS_STATE:
                # Note: The shape here is the base shape (state_dim),
                # not the stacked shape (2, state_dim)
                input_features[key] = PolicyFeature(
                    type=FeatureType.STATE,
                    shape=ft_meta["shape"],
                )
            elif key.startswith("observation.images."):
                input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=ft_meta["shape"],
                )
            elif key.startswith("observation.environment_state"):
                input_features[key] = PolicyFeature(
                    type=FeatureType.ENV,
                    shape=ft_meta["shape"],
                )

    # Build output features from dataset
    output_features = {}
    for key, ft_meta in ds_meta.features.items():
        if key == ACTION:
            output_features[key] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=ft_meta["shape"],
            )

    # Update policy config with features
    cfg.policy.input_features = input_features
    cfg.policy.output_features = output_features
    cfg.policy.device = cfg.device

    logging.info(f"Policy input features: {list(input_features.keys())}")
    logging.info(f"Policy output features: {list(output_features.keys())}")

    return ACTUMIPolicy(cfg.policy)


def update_policy(
    train_metrics: MetricsTracker,
    policy: ACTUMIPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
) -> tuple[MetricsTracker, dict]:
    """Performs a single training step."""
    start_time = time.perf_counter()
    policy.train()

    loss, output_dict = policy.forward(batch)

    loss.backward()

    if grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    optimizer.step()
    optimizer.zero_grad()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    return train_metrics, output_dict


@draccus.wrap()
def train(cfg: TrainACTUMIConfig):
    """Main training function for ACT with Relative Joint Positions (UMI-style)."""

    init_logging()

    logging.info(pformat(cfg))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Auto-select device
    if cfg.device is None:
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"

    device = torch.device(cfg.device)
    logging.info(f"Using device: {device}")

    # Set default job_name if not provided
    if cfg.job_name is None:
        cfg.job_name = f"act_umi_{cfg.dataset.repo_id.split('/')[-1]}"

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    logging.info("Creating dataset")
    dataset = make_dataset_for_umi_policy(cfg)

    # Compute relative statistics for normalization (UMI-style)
    # This computes stats on delta_obs and relative_actions rather than absolute values
    logging.info("Computing relative statistics for normalization...")
    relative_stats = compute_relative_stats(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Create policy
    logging.info("Creating policy")
    policy = make_policy_from_dataset(cfg, dataset.meta)

    # Set relative stats before moving to device (buffers will be registered)
    policy.set_relative_stats(relative_stats)

    policy = policy.to(device)

    # Create preprocessor for training
    preprocessor, postprocessor = make_act_umi_pre_post_processors(
        config=cfg.policy,
        dataset_stats=dataset.meta.stats,
    )

    # Create optimizer and scheduler
    # Match lerobot's ACT training pattern: use policy.get_optim_params() which returns
    # param groups with explicit lr values (base lr for non-backbone, lr_backbone for backbone)
    logging.info("Creating optimizer")
    from torch.optim import AdamW

    # Get param groups from policy (already includes explicit lr for each group)
    param_groups = policy.get_optim_params()

    # Create optimizer - lr and weight_decay here are defaults for groups without explicit values
    # Since get_optim_params() already sets explicit lr for each group, these defaults won't override
    # but are kept for consistency with lerobot's pattern
    optimizer = AdamW(
        param_groups,
        lr=cfg.policy.optimizer_lr,  # Default lr (used if param groups don't specify)
        weight_decay=cfg.policy.optimizer_weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    lr_scheduler = None

    # Handle resume from checkpoint
    start_step = 0
    if cfg.resume:
        last_checkpoint_dir = get_last_checkpoint_dir(cfg.output_dir)
        if last_checkpoint_dir is None:
            raise ValueError(
                f"--resume was set but no checkpoint found in {cfg.output_dir / CHECKPOINTS_DIR}"
            )
        logging.info(f"Resuming from checkpoint: {last_checkpoint_dir}")

        # Load policy weights (includes relative_stats buffers)
        load_policy_weights_from_checkpoint(policy, last_checkpoint_dir)

        # Verify that relative stats were loaded from checkpoint
        if not policy.has_relative_stats:
            logging.warning(
                "Checkpoint does not contain relative stats buffers. "
                "This may be an older checkpoint trained without stats normalization."
            )
            # Set stats from freshly computed values
            policy.set_relative_stats(relative_stats)
            logging.info("Set relative stats from freshly computed values.")
        else:
            # Verify that freshly computed stats match the checkpoint's stats
            # (they should be identical since the dataset hasn't changed)
            checkpoint_stats = {
                "delta_obs": {
                    "mean": policy.delta_obs_normalizer.mean.cpu().numpy(),
                    "std": policy.delta_obs_normalizer.std.cpu().numpy(),
                },
                "relative_action": {
                    "mean": policy.relative_action_normalizer.mean.cpu().numpy(),
                    "std": policy.relative_action_normalizer.std.cpu().numpy(),
                },
            }

            # Compare with freshly computed stats
            tolerance = 1e-5
            for key in ["delta_obs", "relative_action"]:
                for stat in ["mean", "std"]:
                    checkpoint_val = checkpoint_stats[key][stat]
                    computed_val = relative_stats[key][stat]
                    if not np.allclose(
                        checkpoint_val, computed_val, atol=tolerance, rtol=tolerance
                    ):
                        max_diff = np.abs(checkpoint_val - computed_val).max()
                        logging.warning(
                            f"Relative stats mismatch for {key}.{stat}! "
                            f"Max diff: {max_diff:.2e}. This suggests the dataset has changed "
                            f"or stats computation is non-deterministic."
                        )
                        raise ValueError(
                            f"Stats mismatch detected. Checkpoint {key}.{stat} differs from "
                            f"freshly computed stats by up to {max_diff:.2e} (tolerance: {tolerance:.2e})"
                        )

            logging.info("Verified: checkpoint stats match freshly computed stats ✓")

        policy = policy.to(device)  # Move to device after loading weights

        # Load training state (optimizer, rng, step)
        start_step, optimizer, lr_scheduler = load_training_state(
            last_checkpoint_dir, optimizer, lr_scheduler
        )
        logging.info(f"Resumed training from step {start_step}")

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(f"Output dir: {cfg.output_dir}")
    logging.info(f"Steps: {cfg.steps} ({format_big_number(cfg.steps)})")
    logging.info(
        f"Dataset frames: {dataset.num_frames} ({format_big_number(dataset.num_frames)})"
    )
    logging.info(f"Dataset episodes: {dataset.num_episodes}")
    logging.info(f"Batch size: {cfg.batch_size}")
    logging.info(
        f"Learnable params: {num_learnable_params} ({format_big_number(num_learnable_params)})"
    )
    logging.info(
        f"Total params: {num_total_params} ({format_big_number(num_total_params)})"
    )

    # Create dataloader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    dl_iter = cycle(dataloader)

    # Training metrics
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=start_step,
    )

    # Initialize WandB if enabled
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = WandBLogger(cfg)

    logging.info("Starting training")

    for step in range(start_step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.grad_clip_norm,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        train_tracker.step()

        # Logging
        if step % cfg.log_freq == 0:
            logging.info(train_tracker)
            logging.info(f"  l1_loss: {output_dict.get('l1_loss', 0):.4f}")
            if cfg.policy.use_vae:
                logging.info(f"  kld_loss: {output_dict.get('kld_loss', 0):.4f}")

            # Log to WandB
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                # Add extra metrics from output_dict
                if "l1_loss" in output_dict:
                    wandb_log_dict["l1_loss"] = output_dict["l1_loss"]
                if cfg.policy.use_vae and "kld_loss" in output_dict:
                    wandb_log_dict["kld_loss"] = output_dict["kld_loss"]
                wandb_logger.log_dict(wandb_log_dict, step)

            # Reset averages so next log shows stats for the last log_freq steps only
            train_tracker.reset_averages()

        # Checkpointing
        if (step + 1) % cfg.save_freq == 0 or (step + 1) == cfg.steps:
            checkpoint_dir = get_step_checkpoint_dir(
                cfg.output_dir, cfg.steps, step + 1
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            pretrained_dir = checkpoint_dir / "pretrained_model"
            pretrained_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(pretrained_dir)
            # Save optimizer and scheduler state
            save_training_state(checkpoint_dir, step + 1, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            logging.info(f"Saved checkpoint to {checkpoint_dir}")

            # Log checkpoint to WandB
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    logging.info("Training complete!")
    logging.info(f"Final checkpoint saved to {cfg.output_dir}")

    # Push to HuggingFace Hub if enabled
    if hasattr(cfg.policy, "push_to_hub") and cfg.policy.push_to_hub:
        if not hasattr(cfg.policy, "repo_id") or not cfg.policy.repo_id:
            logging.warning(
                "push_to_hub is enabled but repo_id is not set. Skipping push to Hub."
            )
        else:
            logging.info(f"Pushing model to HuggingFace Hub: {cfg.policy.repo_id}")
            try:
                # Get the final checkpoint directory
                final_checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, cfg.steps, cfg.steps
                )
                pretrained_dir = final_checkpoint_dir / "pretrained_model"

                if not pretrained_dir.exists():
                    logging.warning(
                        f"Final checkpoint not found at {pretrained_dir}. "
                        "Saving model before pushing to Hub."
                    )
                    pretrained_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(pretrained_dir)
                    preprocessor.save_pretrained(pretrained_dir)
                    postprocessor.save_pretrained(pretrained_dir)

                # Push model to Hub
                policy.push_to_hub(
                    repo_id=cfg.policy.repo_id,
                    commit_message=f"Training completed after {cfg.steps} steps",
                )
                # Push preprocessor and postprocessor (contain normalization stats)
                preprocessor.push_to_hub(cfg.policy.repo_id)
                postprocessor.push_to_hub(cfg.policy.repo_id)
                logging.info(f"Successfully pushed model to {cfg.policy.repo_id}")
            except Exception as e:
                logging.error(f"Failed to push model to Hub: {e}", exc_info=True)


if __name__ == "__main__":
    train()
