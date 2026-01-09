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

from dataclasses import dataclass
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


@ProcessorStepRegistry.register("image_pad_square_resize_processor")
@dataclass
class ImagePadSquareResizeProcessorStep(ObservationProcessorStep):
    """
    Pads images to square with black borders and resizes to a specified resolution.

    This step applies the following transformations to all image observations:
    1. If downscale_img_square is None, do nothing.
    2. Convert image to float32 if needed.
    3. Pad the image with constant black borders to make it square.
    4. Downscale the image using area interpolation to the desired resolution.

    Attributes:
        downscale_img_square: The target square resolution (e.g., 224 for 224x224).
                              If None, no transformation is applied.
    """

    downscale_img_square: int | None = None

    def observation(self, observation: dict) -> dict:
        """
        Applies padding and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary containing image tensors.
                         Images are expected to have shape (1, C, H, W), (T, C, H, W),
                         or (B, T, C, H, W) after AddBatchDimensionProcessorStep.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.downscale_img_square is None:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]

            # Skip non-tensor values or tensors that don't look like images
            if not isinstance(image, torch.Tensor):
                continue

            original_ndim = len(image.shape)
            # Only process tensors with 4 or 5 dimensions (actual images)
            # Skip 2D/3D tensors that might have "image" in their key name
            if original_ndim not in (4, 5):
                continue

            # Ensure image is float32 tensor (interpolate requires float)
            if image.dtype != torch.float32:
                if image.dtype == torch.uint8:
                    image = image.float() / 255.0
                else:
                    image = image.float()

            device = image.device

            # Normalize to (N, C, H, W) format for processing
            # After AddBatchDimensionProcessorStep, we only see 4D or 5D shapes
            restore_5d = False
            if original_ndim == 4:
                # (1, C, H, W) or (T, C, H, W) -> already in (N, C, H, W) format
                # No reshaping needed
                pass
            elif original_ndim == 5:
                # (B, T, C, H, W) -> flatten to (B*T, C, H, W)
                B, T, C, H, W = image.shape
                image = image.view(B * T, C, H, W)
                restore_5d = True
                original_B, original_T = B, T

            # Now image is (N, C, H, W) where N >= 1
            N, C, H, W = image.shape

            # Pad to square
            max_dim = max(H, W)
            pad_h = max_dim - H
            pad_w = max_dim - W
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

            # Move to CPU for interpolation if on MPS (MPS has issues with some operations)
            is_mps = device.type == "mps"
            if is_mps:
                image = image.cpu()

            # Pad to square
            image = F.pad(image, padding, fill=0, padding_mode="constant")

            # Resize using interpolate
            if self.downscale_img_square > 0:
                image = torch.nn.functional.interpolate(
                    image,
                    size=(self.downscale_img_square, self.downscale_img_square),
                    mode="area",
                )

            # Restore original shape if needed (only for 5D case)
            if restore_5d:
                # Restore (B, T, C, H, W) from (B*T, C, H, W)
                image = image.view(
                    original_B, original_T, C, image.shape[-2], image.shape[-1]
                )

            # Move back to original device
            if is_mps:
                image = image.to(device)

            new_observation[key] = image

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the downscale_img_square parameter.
        """
        return {
            "downscale_img_square": self.downscale_img_square,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.downscale_img_square is None:
            return features

        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(
                        nb_channel,
                        self.downscale_img_square,
                        self.downscale_img_square,
                    ),
                )
        return features


def make_act_relative_rtc_pre_post_processors(
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

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        ImagePadSquareResizeProcessorStep(
            downscale_img_square=config.downscale_img_square
        ),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
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
