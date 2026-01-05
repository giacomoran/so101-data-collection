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
"""Configuration for ACT with Relative Joint Positions.

This configuration extends the standard ACT configuration to support relative
joint positions. The key difference is that observation_delta_indices includes
past frames to fetch obs.state[t-N] for computing the observation delta.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_relative_rtc")
@dataclass
class ACTRelativeRTCConfig(PreTrainedConfig):
    """Configuration class for ACT with Relative Joint Positions.

    This is a modified version of ACTConfig that:
    - Uses relative joint positions as action representation (relative to obs.state[t])
    - Uses observation deltas (obs.state[t] - obs.state[t-N]) as input

    The key change is that observation_delta_indices includes past frames to fetch
    observation states for computing deltas.

    Notes on the inputs and outputs:
        - At least one key starting with "observation.image" is required as input AND/OR
          the key "observation.environment_state" is required as input.
        - "observation.state" is REQUIRED for this policy (to compute relative positions).
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        obs_state_delta_frames: Number of frames back for computing observation state delta.
        use_rtc: Enable Real-Time Chunking (training-time smoothing). Reserved for future use.
        ... (other args same as ACTConfig)
    """

    # Input / output structure.
    n_obs_steps: int = 2  # Kept for backward compatibility, but see obs_state_delta_frames
    chunk_size: int = 100
    n_action_steps: int = 100

    # Number of frames back for computing observation state delta.
    # delta_obs = obs.state[t] - obs.state[t - obs_state_delta_frames]
    # Default=1 means delta over 1 frame (33ms at 30Hz).
    # Use obs_state_delta_frames=3 for delta over 3 frames (100ms at 30Hz, ~10Hz effective).
    # Higher values give larger, more meaningful deltas when data is high-frequency.
    obs_state_delta_frames: int = 1

    # Real-Time Chunking (RTC) - reserved for future implementation
    # When enabled, will apply training-time smoothing to reduce discontinuities at chunk boundaries
    use_rtc: bool = False

    # Normalization configuration for ACT Relative:
    #
    # We disable normalization for STATE and ACTION features because the model computes
    # relative transformations from absolute values:
    #   - Observation deltas: delta_obs = obs[t] - obs[t-N]
    #   - Relative actions: relative_action = action - obs[t]
    #
    # Normalization must happen AFTER computing these relative values, not before, because:
    #     normalize(a) - normalize(b) ≠ normalize(a - b)
    #
    # If we normalized absolute values first, the relative transformations would be computed
    # on normalized data, which has different statistical properties than the true relative
    # values. The distributions of absolute positions (mean≈0.5rad, std≈1.0rad) differ
    # fundamentally from relative changes (mean≈0.0rad, std≈0.05rad).
    #
    # Images are still normalized because they don't undergo relative transformations and
    # normalization is essential for vision backbones trained on ImageNet statistics.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,  # Keep normalization for images
            "STATE": NormalizationMode.IDENTITY,  # Disable: we compute deltas from absolute values
            "ACTION": NormalizationMode.IDENTITY,  # Disable: we compute relative actions from absolute values
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )
        # For relative positions, we need the robot state
        if not self.robot_state_feature:
            raise ValueError(
                "ACTRelativeRTCConfig requires 'observation.state' as input for computing relative positions."
            )

    @property
    def state_delta_indices(self) -> list[int]:
        """Return indices for fetching obs.state[t-N] and obs.state[t].

        Uses obs_state_delta_frames to determine N.
        Example with obs_state_delta_frames=3: returns [-3, 0]

        The dataset will return observation.state with shape [2, state_dim]:
        - observation.state[0] = obs.state[t - obs_state_delta_frames]
        - observation.state[1] = obs.state[t]
        """
        return [-self.obs_state_delta_frames, 0]

    @property
    def image_delta_indices(self) -> list[int]:
        """Return indices for fetching images.

        NOTE: Due to lerobot's resolve_delta_timestamps() applying observation_delta_indices
        uniformly to ALL observation features, images will also be loaded with state_delta_indices.
        This means 2 image frames are loaded when only 1 is needed. The model uses only the
        last frame (index -1 or 1). This is wasteful but functional.

        See README.md for details on this lerobot limitation.
        """
        return [0]

    @property
    def observation_delta_indices(self) -> list[int]:
        """Return indices applied to ALL observation features by lerobot.

        This returns state_delta_indices because lerobot's resolve_delta_timestamps()
        applies the same indices to all observation.* features. Since we need multiple
        frames for state delta computation, images will also load multiple frames
        (wasteful - we only use the last one).

        Note: Use state_delta_indices and image_delta_indices for internal logic.
        """
        return self.state_delta_indices

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

