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
        --batch_size=8
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import torch
from lerobot.configs.default import DatasetConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import cycle
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging
from torch.optim import Optimizer

# Import our ACT UMI policy
from act_umi import ACTUMIConfig, ACTUMIPolicy, make_act_umi_pre_post_processors


@dataclass
class TrainACTUMIConfig:
    """Configuration for training ACT with Relative Joint Positions (UMI-style)."""

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Policy configuration (will be populated from dataset)
    policy: ACTUMIConfig = field(default_factory=ACTUMIConfig)

    # Training configuration
    steps: int = 50000
    batch_size: int = 8
    num_workers: int = 4
    seed: int | None = 42

    # Optimizer configuration
    grad_clip_norm: float = 10.0

    # Logging and checkpointing
    output_dir: Path = Path("outputs/train_act_umi")
    log_freq: int = 100
    save_freq: int = 5000

    # Tolerance for timestamp matching
    tolerance_s: float = 1e-4

    # Device
    device: str | None = None


def resolve_delta_timestamps_for_umi(
    config: ACTUMIConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list[float]]:
    """Resolves delta_timestamps for ACTUMIPolicy.

    Returns delta_timestamps that include:
    - observation.state: [-1/fps, 0] to get obs.state[t-1] and obs.state[t]
    - action: [0, 1/fps, 2/fps, ...] for the action chunk
    """
    delta_timestamps = {}

    for key in ds_meta.features:
        if key == ACTION and config.action_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in config.action_delta_indices
            ]
        if key.startswith(OBS_PREFIX) and config.observation_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in config.observation_delta_indices
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
    """Creates an ACTUMIPolicy configured from dataset metadata."""

    # Build input features from dataset
    input_features = {}
    output_features = {}

    for key, ft_meta in ds_meta.features.items():
        if key == ACTION:
            output_features[key] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=ft_meta["shape"],
            )
        elif key == OBS_STATE:
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

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    logging.info("Creating dataset")
    dataset = make_dataset_for_umi_policy(cfg)

    # Create policy
    logging.info("Creating policy")
    policy = make_policy_from_dataset(cfg, dataset.meta)
    policy = policy.to(device)

    # Create preprocessor for training
    preprocessor, postprocessor = make_act_umi_pre_post_processors(
        config=cfg.policy,
        dataset_stats=dataset.meta.stats,
    )

    # Create optimizer and scheduler
    logging.info("Creating optimizer")
    from torch.optim import AdamW

    optimizer = AdamW(
        policy.get_optim_params(),
        lr=cfg.policy.optimizer_lr,
        weight_decay=cfg.policy.optimizer_weight_decay,
    )
    lr_scheduler = None

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
        initial_step=0,
    )

    logging.info("Starting training")

    for step in range(cfg.steps):
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
            if cfg.policy.use_vae:
                logging.info(f"  kld_loss: {output_dict.get('kld_loss', 0):.4f}")

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

    logging.info("Training complete!")
    logging.info(f"Final checkpoint saved to {cfg.output_dir}")


if __name__ == "__main__":
    train()
