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

    # Cached padding parameters (computed on first call)
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

            # Short-circuit if already at target resolution (avoids CPU work during training
            # when dataset images are already preprocessed to target size)
            if H == self.target_resolution and W == self.target_resolution:
                continue

            # Cache padding parameters on first call
            # Order: pad to square FIRST, then scale (matches ffmpeg preprocessing)
            if self._padding is None:
                max_dim = max(H, W)
                h_pad = max_dim - H
                w_pad = max_dim - W
                # torchvision.transforms.functional.pad uses (left, top, right, bottom)
                self._padding = (w_pad // 2, h_pad // 2, w_pad - w_pad // 2, h_pad - h_pad // 2)

            # Pad to square first (matches ffmpeg: pad=max(iw,ih):max(iw,ih):...)
            image = F.pad(image, self._padding, fill=0, padding_mode="constant")

            # Then scale to target resolution (matches ffmpeg: scale=target:target)
            # Use mode="area" for better ffmpeg matching. On MPS, area mode has a limitation
            # (input must be divisible by output), so we transfer to CPU for that operation.
            if image.device.type == "mps":
                image_cpu = image.cpu()
                image_cpu = torch.nn.functional.interpolate(
                    image_cpu, size=(self.target_resolution, self.target_resolution), mode="area"
                )
                image = image_cpu.to("mps")
            else:
                image = torch.nn.functional.interpolate(
                    image, size=(self.target_resolution, self.target_resolution), mode="area"
                )

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
    ]

    # Add resize step if downscale_img_square is set.
    # The step short-circuits when images are already at target resolution (no CPU overhead).
    # IMPORTANT: Pad/resize BEFORE normalization so that fill=0 gives true black pixels.
    if config.downscale_img_square is not None:
        keys_image = [k for k in config.input_features if "image" in k]
        input_steps.append(
            ImagePadSquareResizeProcessorStep(
                target_resolution=config.downscale_img_square,
                keys_image=keys_image,
            ),
        )

    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
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
