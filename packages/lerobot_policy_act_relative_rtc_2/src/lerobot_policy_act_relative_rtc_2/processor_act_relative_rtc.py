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
"""Processor for ACT with Relative Joint Positions.

This processor is largely the same as the standard ACT processor.
The relative position transformations are handled in the model's forward pass,
not in the processor.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torchvision.transforms.functional as F
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_act_relative_rtc import ACTRelativeRTCConfig


@ProcessorStepRegistry.register("image_pad_square_resize_processor_2")
@dataclass
class ImagePadSquareResizeProcessorStep(ObservationProcessorStep):
    """Pads images to square with black borders and resizes to a specified resolution.

    This step is only added to the pipeline when resizing is actually needed
    (determined at pipeline construction time by comparing downscale_img_square
    with dataset image dimensions).

    Attributes:
        target_resolution: The target square resolution (e.g., 224 for 224x224).
        keys_image: List of image keys to process.
        downscale_img_square: (Deprecated) Alias for target_resolution for backward compatibility.
    """

    target_resolution: int = 224
    keys_image: list[str] = field(default_factory=list)
    downscale_img_square: int | None = None  # Backward compatibility

    # Cached resize parameters (computed on first call)
    _hw_resize: tuple[int, int] | None = None
    _padding: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        """Handle backward compatibility with old parameter name."""
        if self.downscale_img_square is not None:
            self.target_resolution = self.downscale_img_square

    def observation(self, observation: dict) -> dict:
        new_observation = dict(observation)

        for key in self.keys_image:
            image = observation[key]

            # Flatten 5D (B, T, C, H, W) to 4D (B*T, C, H, W) for processing
            shape_original = image.shape
            if len(shape_original) == 5:
                B, T, C, H, W = shape_original
                image = image.view(B * T, C, H, W)
            else:
                _, C, H, W = shape_original

            # Cache resize parameters on first call
            if self._hw_resize is None:
                scale = self.target_resolution / max(H, W)
                h_new, w_new = int(H * scale), int(W * scale)
                self._hw_resize = (h_new, w_new)

                h_pad = self.target_resolution - h_new
                w_pad = self.target_resolution - w_new
                self._padding = (w_pad // 2, h_pad // 2, w_pad - w_pad // 2, h_pad - h_pad // 2)

            # Resize (preserving aspect ratio)
            image = torch.nn.functional.interpolate(image, size=self._hw_resize, mode="bilinear", align_corners=False)

            # Pad to square
            image = F.pad(image, self._padding, fill=0, padding_mode="constant")

            # Restore 5D shape if needed
            if len(shape_original) == 5:
                image = image.view(B, T, C, self.target_resolution, self.target_resolution)

            new_observation[key] = image

        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "target_resolution": self.target_resolution,
            "keys_image": self.keys_image,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for key in self.keys_image:
            if key in features[PipelineFeatureType.OBSERVATION]:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, self.target_resolution, self.target_resolution),
                )
        return features


def _get_keys_image_needing_resize(
    config: ACTRelativeRTCConfig,
) -> list[str]:
    """Get list of image keys that need resizing.

    Returns empty list if:
    - downscale_img_square is None
    - All images are already at target resolution

    Args:
        config: Policy config with input_features and downscale_img_square.

    Returns:
        List of image feature keys that need resizing.
    """
    if config.downscale_img_square is None:
        return []

    target = config.downscale_img_square
    keys_needing_resize = []

    for key, feature in config.input_features.items():
        if "image" not in key:
            continue
        # Feature shape is (C, H, W)
        _, H, W = feature.shape
        if H != target or W != target:
            keys_needing_resize.append(key)

    return keys_needing_resize


def make_act_relative_rtc_2_pre_post_processors(
    config: ACTRelativeRTCConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT Relative RTC policy.

    The processing is the same as standard ACT - the relative position transformations
    are handled in the model's forward pass, not here.

    Note: This function name follows lerobot's naming convention for dynamic import:
    make_{policy_type}_pre_post_processors

    Args:
        config (ACTRelativeRTCConfig): The policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): Dataset statistics for normalization.

    Returns:
        tuple: A tuple containing the pre-processor pipeline and the post-processor pipeline.
    """

    assert config.device is not None

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    # Only add image resize step if images actually need resizing.
    # Skip if: downscale_img_square is None, or images are already at target resolution.
    keys_image = _get_keys_image_needing_resize(config)
    if keys_image:
        input_steps.insert(
            4,  # After DeviceProcessorStep (adjusted index due to drop step above)
            ImagePadSquareResizeProcessorStep(
                target_resolution=config.downscale_img_square,
                keys_image=keys_image,
            ),
        )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
