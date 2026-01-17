#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Modified 2025 by Giacomo Randazzo for ACT with relative joint positions.
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
"""ACT Policy with Relative Joint Positions.

This is a modified version of ACT that:
- Uses relative joint positions as action representation (action - obs.state[t])
- Uses observation deltas (obs.state[t] - obs.state[t-N]) as input

Key changes from standard ACT:
- forward(): Converts absolute actions to relative before computing loss
- forward(): Computes observation delta from obs.state[t-N] and obs.state[t]
- select_action(): Converts predicted relative actions back to absolute
- Uses queue-based observation history tracking (following lerobot conventions)

Supports precomputed relative stats from meta/relative_stats.json (created by
preprocess_dataset.py) to skip the ~2min stats computation at training init.
"""

import json
import logging
import math
from collections import deque
from collections.abc import Callable
from itertools import chain
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act_relative_rtc import ACTRelativeRTCConfig

# Custom constant for RTC delays (not in lerobot.utils.constants)
DELAYS = "delays"


class Normalizer(nn.Module):
    """Affine normalization module with persistent mean/std buffers.

    This is the idiomatic PyTorch pattern for normalization statistics:
    - Buffers are always registered in __init__ with identity transform defaults
    - They serialize properly to state_dict (no None values)
    - No custom load_state_dict() needed
    - Clean encapsulation of normalize/unnormalize operations
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Always register buffers with identity transform defaults (mean=0, std=1)
        # This ensures they're always in state_dict and serialize properly
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float32))
        # Track whether stats have been configured (also serialized)
        self.register_buffer("_is_configured", torch.tensor(False))

    @property
    def is_configured(self) -> bool:
        """Check if normalization statistics have been set."""
        return self._is_configured.item()

    def configure(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set normalization statistics. Call once after computing stats from data."""
        self.mean.copy_(mean.to(dtype=torch.float32))
        self.std.copy_(std.to(dtype=torch.float32))
        self._is_configured.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize: (x - mean) / (std + eps)"""
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: Tensor) -> Tensor:
        """Unnormalize: x * std + mean"""
        return x * self.std + self.mean


class ACTRelativeRTCPolicy(PreTrainedPolicy):
    """ACT Policy with Relative Joint Positions.

    This policy predicts actions relative to the current observation state,
    and uses observation deltas as input instead of absolute positions.

    Training:
        - Receives obs.state with shape [batch, 2, state_dim] containing [obs[t-N], obs[t]]
        - Computes delta_obs = obs[t] - obs[t-N]
        - Converts absolute actions to relative: rel_action = action - obs[t]
        - Predicts relative actions and computes loss

    Inference:
        - Maintains observation history using a queue (following lerobot conventions)
        - Computes delta_obs = obs[t] - obs[t-N] from stacked observations
        - Predicts relative actions
        - Converts back to absolute: action = rel_action + obs[t]
    """

    config_class = ACTRelativeRTCConfig
    name = "act_relative_rtc_2"

    def __init__(
        self,
        config: ACTRelativeRTCConfig,
        dataset_meta=None,
        **kwargs,
    ):
        """Initialize the ACT Relative RTC policy.

        Args:
            config: Policy configuration.
            dataset_meta: Optional LeRobotDatasetMetadata. If provided and relative stats
            are not already configured (e.g., from a checkpoint), the dataset will be
            recreated and relative stats will be computed. This happens automatically
            when using lerobot-train.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTRelativeRTC(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        # Normalizer for relative actions (V2: removed delta_obs_normalizer)
        # This is a proper nn.Module with persistent buffers that serialize correctly.
        # It starts with identity transform (mean=0, std=1) and is configured via
        # set_relative_stats() or loaded from checkpoint automatically.
        action_dim = config.action_feature.shape[0]
        self.relative_action_normalizer = Normalizer(action_dim)

        self.reset()

        # Load or compute relative stats from dataset if provided and not already configured.
        # This happens when training from scratch with lerobot-train.
        # When loading from pretrained (e.g., lerobot-record for evaluation), stats are
        # loaded from the checkpoint AFTER __init__, so we skip computing here.
        # We detect this by checking if pretrained_path is set in config.
        # Can also be skipped via skip_compute_relative_stats flag (useful for debugging).
        if (
            dataset_meta is not None
            and not self.has_relative_stats
            and self.config.pretrained_path is None
            and not self.config.skip_compute_relative_stats
        ):
            # Try to load precomputed stats from meta/relative_stats.json first
            if not self._try_load_precomputed_relative_stats(dataset_meta):
                # Fall back to computing stats from the dataset
                self._compute_and_set_relative_stats(dataset_meta)

    def _try_load_precomputed_relative_stats(self, dataset_meta) -> bool:
        """Try to load precomputed relative stats from meta/relative_stats.json.

        Precomputed stats can be created using preprocess_dataset.py, which
        eliminates the ~2min stats computation at every training init.

        Args:
            dataset_meta: LeRobotDatasetMetadata with root path.

        Returns:
            True if precomputed stats were found and loaded, False otherwise.
        """
        path_relative_stats = Path(dataset_meta.root) / "meta" / "relative_stats.json"

        if not path_relative_stats.exists():
            return False

        logging.info(f"Loading precomputed relative stats from {path_relative_stats}")

        with open(path_relative_stats) as f:
            stats_json = json.load(f)

        # Validate config compatibility - these MUST match or stats are invalid
        config_stored = stats_json.get("config", {})
        obs_delta_frames_stored = config_stored.get("obs_state_delta_frames")
        chunk_size_stored = config_stored.get("chunk_size")

        if obs_delta_frames_stored is not None:
            if obs_delta_frames_stored != self.config.obs_state_delta_frames:
                raise ValueError(
                    f"Precomputed stats obs_state_delta_frames ({obs_delta_frames_stored}) "
                    f"differs from training config ({self.config.obs_state_delta_frames}). "
                    "Re-run preprocess_dataset.py with matching --obs-state-delta-frames."
                )

        if chunk_size_stored is not None:
            if chunk_size_stored != self.config.chunk_size:
                raise ValueError(
                    f"Precomputed stats chunk_size ({chunk_size_stored}) "
                    f"differs from training config ({self.config.chunk_size}). "
                    "Re-run preprocess_dataset.py with matching --chunk-size."
                )

        # Convert lists to numpy arrays and set stats
        stats = {
            "delta_obs": {
                "mean": np.array(stats_json["delta_obs"]["mean"], dtype=np.float32),
                "std": np.array(stats_json["delta_obs"]["std"], dtype=np.float32),
            },
            "relative_action": {
                "mean": np.array(stats_json["relative_action"]["mean"], dtype=np.float32),
                "std": np.array(stats_json["relative_action"]["std"], dtype=np.float32),
            },
        }

        self.set_relative_stats(stats)
        logging.info("Precomputed relative stats loaded successfully.")
        return True

    def _compute_and_set_relative_stats(self, dataset_meta) -> None:
        """Compute relative stats from dataset metadata and configure normalizers.

        This method recreates the dataset using dataset_meta.repo_id and dataset_meta.root,
        then iterates through it to compute statistics on relative values (delta_obs and
        relative_action). This is a one-time cost during initial training.

        Note: This recreates the dataset, which is redundant with lerobot's dataset creation,
        but unavoidable given lerobot's API (make_policy only receives metadata, not the dataset).

        Args:
            dataset_meta: LeRobotDatasetMetadata with repo_id and root for dataset recreation.
        """
        from .relative_stats import compute_relative_stats

        logging.info("Computing relative stats from dataset (one-time initialization)...")

        # Recreate the dataset with appropriate delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Build delta_timestamps for relative stats computation
        delta_timestamps = {}
        for key in dataset_meta.features:
            if key == ACTION:
                delta_timestamps[key] = [i / dataset_meta.fps for i in self.config.action_delta_indices]
            elif key == OBS_STATE:
                delta_timestamps[key] = [i / dataset_meta.fps for i in self.config.state_delta_indices]
            elif key.startswith("observation."):
                # For images and other observations, use state_delta_indices
                # (lerobot applies same indices to all observations)
                delta_timestamps[key] = [i / dataset_meta.fps for i in self.config.state_delta_indices]

        dataset = LeRobotDataset(
            dataset_meta.repo_id,
            root=dataset_meta.root,
            delta_timestamps=delta_timestamps,
        )

        # Compute relative stats
        stats = compute_relative_stats(
            dataset,
            batch_size=64,
            num_workers=self.config.num_workers,
        )

        # Configure normalizers
        self.set_relative_stats(stats)
        logging.info("Relative stats computed and configured successfully.")

    def get_optim_params(self) -> list[dict]:
        """Return parameter groups for optimizer with differential learning rates.

        Matches lerobot's ACT policy implementation:
        - First group: all params except backbone, uses optimizer_lr
        - Second group: backbone params only, uses optimizer_lr_backbone
        """
        return [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    @property
    def has_relative_stats(self) -> bool:
        """Check if relative stats normalizers have been configured.

        V2 change: Only checks relative_action_normalizer (delta_obs_normalizer removed).
        """
        return self.relative_action_normalizer.is_configured

    def set_relative_stats(self, stats: dict) -> None:
        """Configure normalizers with computed statistics.

        V2 change: Only configures relative_action_normalizer (delta_obs removed).

        These stats are used to normalize relative actions in the forward pass,
        normalizing relative values rather than absolute values.

        Args:
            stats: Dictionary with structure:
                {
                    "relative_action": {"mean": np.array, "std": np.array}
                }
        """
        self.relative_action_normalizer.configure(
            mean=torch.as_tensor(stats["relative_action"]["mean"]),
            std=torch.as_tensor(stats["relative_action"]["std"]),
        )

    def reset(self):
        """This should be called whenever the environment is reset.

        V2 change: Removed observation queue (no longer needed).
        """
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        V2 changes:
        - No observation queue (removed delta_obs computation)
        - Get obs_state_t directly from batch

        Note: This method does not support RTC (action prefix conditioning).
        For RTC inference, use predict_action_chunk directly with action_prefix.

        Requires relative stats to be configured (via checkpoint or call to set_relative_stats()).
        """
        if not self.has_relative_stats:
            raise RuntimeError(
                "select_action() requires relative stats. Load from checkpoint or call set_relative_stats()."
            )
        self.eval()

        # Get obs_state_t from batch - squeeze temporal dim if present (from dataloader)
        obs_state_t = batch[OBS_STATE]
        if obs_state_t.dim() == 3:
            obs_state_t = obs_state_t.squeeze(1)  # [batch, state_dim]

        if self.config.temporal_ensemble_coeff is not None:
            # Predict (normalized) relative actions (no RTC prefix)
            relative_actions_normalized = self.predict_action_chunk(batch)
            # Unnormalize
            relative_actions = self.relative_action_normalizer.inverse(relative_actions_normalized)
            # Convert to absolute
            absolute_actions = relative_actions + obs_state_t.unsqueeze(1)
            action = self.temporal_ensembler.update(absolute_actions)
        else:
            if len(self._action_queue) == 0:
                # Predict (normalized) relative actions (no RTC prefix)
                relative_actions_normalized = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                # Unnormalize
                relative_actions = self.relative_action_normalizer.inverse(relative_actions_normalized)
                # Convert to absolute
                absolute_actions = relative_actions + obs_state_t.unsqueeze(1)
                self._action_queue.extend(absolute_actions.transpose(0, 1))
            action = self._action_queue.popleft()

        return action

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        delay: int = 0,
        action_prefix: Tensor | None = None,
    ) -> Tensor:
        """Predict a chunk of relative actions given observations.

        Args:
            batch: Dictionary containing observations (images, env_state, OBS_STATE).
            delay: Number of action prefix steps to condition on (RTC). If action_prefix is
                provided, its second dimension must equal delay.
            action_prefix: Absolute action prefix (NOT relative), shape [batch, delay, action_dim].
                Will be converted to relative and normalized internally.

        Returns:
            Predicted relative actions (normalized)
        """
        # Set model to evaluation mode
        self.eval()

        # Get obs_state from batch - squeeze temporal dim if present (from dataloader)
        obs_state = batch[OBS_STATE]
        if obs_state.dim() == 3:
            obs_state = obs_state.squeeze(1)  # [batch, state_dim]
        batch_size = obs_state.shape[0]
        max_delay = self.config.rtc_max_delay

        # Shallow copy so we can modify without affecting original
        batch = dict(batch)

        if self.config.image_features:
            # Images should have shape [batch, channels, height, width]
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        if action_prefix is not None and delay > 0:
            assert action_prefix.shape[1] == delay, (
                f"action_prefix.shape[1]={action_prefix.shape[1]} must equal delay={delay}"
            )

            # Convert absolute action_prefix to relative
            relative_action_prefix = action_prefix - obs_state.unsqueeze(1)  # [batch, delay, action_dim]

            # Normalize
            relative_action_prefix = self.relative_action_normalizer(relative_action_prefix)

            # Pad action_prefix to rtc_max_delay
            action_prefix_padded = torch.zeros(
                (batch_size, max_delay, relative_action_prefix.shape[2]),
                dtype=relative_action_prefix.dtype,
                device=relative_action_prefix.device,
            )
            action_prefix_padded[:, :delay] = relative_action_prefix
            delays = torch.full((batch_size,), delay, dtype=torch.long, device=obs_state.device)
        else:
            # No action prefix - create empty prefix with delay=0
            action_dim = self.config.action_feature.shape[0]
            action_prefix_padded = torch.zeros(
                (batch_size, max_delay, action_dim),
                dtype=obs_state.dtype,
                device=obs_state.device,
            )
            delays = torch.zeros(batch_size, dtype=torch.long, device=obs_state.device)

        # Populate batch with ACTION (padded prefix) and DELAYS
        batch[ACTION] = action_prefix_padded
        batch[DELAYS] = delays

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Input batch format (from dataloader):
        - batch[OBS_STATE]: [batch, state_dim] - current observation state
        - batch[ACTION]: [batch, rtc_max_delay + chunk_size, action_dim] - absolute actions
        - batch["action_is_pad"]: [batch, rtc_max_delay + chunk_size] - padding mask

        This method:
        1. Converts absolute actions to normalized relative actions
        2. Samples per-sample delays from {0, ..., rtc_max_delay}
        3. Passes full sequence to inner model (which extracts prefix/targets and masks)
        4. Computes L1 loss on targets only
        """
        # Get obs_state_t - squeeze temporal dim if present (observation_delta_indices=[0] adds dim)
        obs_state_t = batch[OBS_STATE]
        if obs_state_t.dim() == 3:
            obs_state_t = obs_state_t.squeeze(1)  # [batch, state_dim]
        max_delay = self.config.rtc_max_delay
        chunk_size = self.config.chunk_size

        # Validate input shapes
        expected_action_len = max_delay + chunk_size
        actual_action_len = batch[ACTION].shape[1]
        assert actual_action_len == expected_action_len, (
            f"batch[ACTION].shape[1]={actual_action_len} must equal "
            f"rtc_max_delay + chunk_size = {max_delay} + {chunk_size} = {expected_action_len}. "
            f"Check that your training config's chunk_size matches and that the dataset isn't cached "
            f"with old delta_timestamps. Expected action_delta_indices: {self.config.action_delta_indices}"
        )

        # Convert absolute actions to relative actions
        absolute_actions = batch[ACTION]  # [batch, max_delay + chunk_size, action_dim]
        relative_actions = absolute_actions - obs_state_t.unsqueeze(1)

        # Normalize relative actions
        relative_actions = self.relative_action_normalizer(relative_actions)

        # Sample per-sample delays from {0, ..., max_delay}
        batch_size = relative_actions.shape[0]
        delays = torch.randint(0, max_delay + 1, (batch_size,), device=relative_actions.device)

        # Create training batch with full normalized relative sequence and delays
        # The inner model will extract prefix/targets and do masking
        training_batch = dict(batch)
        training_batch[ACTION] = relative_actions  # [batch, max_delay + chunk_size, action_dim]
        training_batch[DELAYS] = delays

        if self.config.image_features:
            training_batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        pred_relative_actions, (mu_hat, log_sigma_x2_hat) = self.model(training_batch)

        # Extract targets for loss computation
        targets = relative_actions[:, max_delay : max_delay + chunk_size]  # [batch, chunk_size, action_dim]

        # Extract padding mask for targets only
        action_is_pad_extended = batch["action_is_pad"]  # [batch, max_delay + chunk_size]
        action_is_pad = action_is_pad_extended[:, max_delay : max_delay + chunk_size]  # [batch, chunk_size]

        # Compute L1 loss on targets (excluding padded actions)
        pad_mask = ~action_is_pad.unsqueeze(-1)
        l1_loss_full = F.l1_loss(targets, pred_relative_actions, reduction="none")
        l1_loss = (l1_loss_full * pad_mask).mean()

        loss_dict = {"l1_loss": l1_loss.item()}

        # Track average delay for RTC training
        loss_dict["rtc_avg_delay"] = delays.float().mean().item()

        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1),
                dtype=torch.long,
                device=self.ensembled_actions.device,
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones_like(self.ensembled_actions_count[-1:]),
                ]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACTRelativeRTC(nn.Module):
    """ACT model modified for relative joint positions.

    The core architecture is identical to ACT, but it operates on:
    - Delta observation states instead of absolute states
    - Predicts relative actions instead of absolute actions
    """

    def __init__(self, config: ACTRelativeRTCConfig):
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # V2 change: Removed vae_encoder_robot_state_input_proj (no robot state in VAE encoder)
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # V2 change: num_input_token_encoder = 1 (cls) + chunk_size (actions), no robot_state
            num_input_token_encoder = 1 + config.chunk_size
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # V2 change: Removed encoder_robot_state_input_proj (no robot state in encoder)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
        # V2 change: n_1d_tokens = 1 (latent) + 1 (env_state if present), no robot_state
        n_1d_tokens = 1  # for the latent
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # RTC components for action prefix conditioning
        action_dim = config.action_feature.shape[0]
        self.action_prefix_proj = nn.Linear(action_dim, config.dim_model)
        self.pad_embed = nn.Parameter(torch.zeros(config.dim_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.action_prefix_proj.weight)
        nn.init.zeros_(self.action_prefix_proj.bias)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through ACT with RTC (Real-Time Chunking).

        Input batch format:
        - Training: batch[ACTION] has shape [batch, max_delay + chunk_size, action_dim]
          (normalized relative actions: prefix + targets)
        - Inference: batch[ACTION] has shape [batch, max_delay, action_dim]
          (normalized relative action prefix only)
        - batch[DELAYS]: [batch] - delay per sample (0 to max_delay)
        - batch[OBS_IMAGES]: list of image tensors (if using images)
        - batch[OBS_ENV_STATE]: environment state (if present)

        The model distinguishes training vs inference by the action sequence length.
        """
        max_delay = self.config.rtc_max_delay
        chunk_size = self.config.chunk_size

        # Validate required batch keys
        assert ACTION in batch, f"batch must contain {ACTION}"
        assert DELAYS in batch, f"batch must contain {DELAYS}"

        actions = batch[ACTION]
        delays = batch[DELAYS]
        action_seq_len = actions.shape[1]

        # Determine if training or inference based on action sequence length
        is_training_batch = action_seq_len == max_delay + chunk_size
        is_inference_batch = action_seq_len == max_delay

        assert is_training_batch or is_inference_batch, (
            f"batch[ACTION].shape[1]={action_seq_len} must be either "
            f"max_delay + chunk_size = {max_delay + chunk_size} (training) or "
            f"max_delay = {max_delay} (inference)"
        )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # VAE encoder (training only)
        if self.config.use_vae and is_training_batch and self.training:
            # Extract targets from the full action sequence
            action_targets = actions[:, max_delay : max_delay + chunk_size]
            action_is_pad_targets = batch["action_is_pad"][:, max_delay : max_delay + chunk_size]

            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            action_embed = self.vae_encoder_action_input_proj(action_targets)

            vae_encoder_input = torch.cat([cls_embed, action_embed], axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 1),
                False,
                device=vae_encoder_input.device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, action_is_pad_targets], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            device = batch[OBS_ENV_STATE].device if OBS_ENV_STATE in batch else batch[OBS_IMAGES][0].device
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # RTC: Process action prefix
        # Extract prefix from batch[ACTION]
        action_prefix = actions[:, :max_delay]  # [batch, max_delay, action_dim]

        if max_delay > 0:
            action_prefix_embed = self.action_prefix_proj(action_prefix)

            # Mask positions >= delay with learnable pad_embed
            mask = torch.arange(max_delay, device=delays.device)[None, :] >= delays[:, None]
            action_prefix_embed = torch.where(
                mask.unsqueeze(-1),
                self.pad_embed[None, None, :].expand_as(action_prefix_embed),
                action_prefix_embed,
            )
            encoder_in_tokens.extend(list(action_prefix_embed.permute(1, 0, 2)))

            # Create positional embeddings for prefix tokens
            num_positions_offset = len(encoder_in_pos_embed)
            prefix_pos_embed = create_sinusoidal_pos_embedding(num_positions_offset + max_delay, self.config.dim_model)[
                num_positions_offset:
            ].to(device=action_prefix.device, dtype=encoder_in_tokens[0].dtype)
            encoder_in_pos_embed.extend(list(prefix_pos_embed.unsqueeze(1)))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTRelativeRTCConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTRelativeRTCConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTRelativeRTCConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTRelativeRTCConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
