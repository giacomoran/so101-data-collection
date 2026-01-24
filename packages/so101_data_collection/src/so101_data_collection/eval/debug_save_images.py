"""Debug utilities for saving inference images.

Saves images at various stages of the preprocessing pipeline for comparison
with training images.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ImageNet normalization stats (default)
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


class InferenceImageSaver:
    """Saves inference images for debugging preprocessing consistency."""

    def __init__(
        self,
        path_output: Path,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        count_max: int = 10,
    ):
        """
        Args:
            path_output: Directory to save images
            mean: Normalization mean (defaults to ImageNet)
            std: Normalization std (defaults to ImageNet)
            count_max: Maximum number of inference frames to save (to avoid filling disk)
        """
        self.path_output = Path(path_output)
        self.path_output.mkdir(parents=True, exist_ok=True)
        self.mean = torch.tensor(mean or MEAN_IMAGENET).view(3, 1, 1)
        self.std = torch.tensor(std or STD_IMAGENET).view(3, 1, 1)
        self.count_max = count_max
        self.count_saved = 0

    def save(
        self,
        idx_chunk: int,
        observation_raw: dict,
        observation_preprocessed: dict,
        camera_names: list[str],
    ) -> None:
        """Save images from a single inference frame.

        Args:
            idx_chunk: Inference chunk index (for filename)
            observation_raw: Raw observation dict from robot (numpy arrays, HWC, 0-255)
            observation_preprocessed: Preprocessed observation (tensors, CHW, normalized)
            camera_names: List of camera names to save
        """
        if self.count_saved >= self.count_max:
            return

        for cam_name in camera_names:
            key_image = f"observation.images.{cam_name}"

            # Save raw image (from robot, numpy HWC 0-255)
            if cam_name in observation_raw:
                img_raw = observation_raw[cam_name]
                if isinstance(img_raw, np.ndarray):
                    Image.fromarray(img_raw).save(self.path_output / f"chunk{idx_chunk:04d}_{cam_name}_1_raw.png")

            # Save preprocessed image (what model sees, tensor CHW normalized)
            if key_image in observation_preprocessed:
                tensor_img = observation_preprocessed[key_image]

                # Handle batch dimension if present
                if tensor_img.dim() == 4:
                    tensor_img = tensor_img[0]  # Take first batch item

                # Denormalize for visual inspection
                img_normalized = tensor_img.cpu()
                img_denorm = img_normalized * self.std + self.mean
                img_denorm_np = (img_denorm.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img_denorm_np).save(self.path_output / f"chunk{idx_chunk:04d}_{cam_name}_2_denorm.png")

        self.count_saved += 1
