#!/usr/bin/env python
"""Debug script for ACTRelativeRTCPolicy with real data.

Run with:
    python debug_act_relative_rtc.py

Or with debugger:
    python -m pdb debug_act_relative_rtc.py
"""

from pathlib import Path

import numpy as np
import torch
from icecream import ic
from PIL import Image
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.random_utils import set_seed
from lerobot_policy_act_relative_rtc_2.configuration_act_relative_rtc import ACTRelativeRTCConfig
from lerobot_policy_act_relative_rtc_2.modeling_act_relative_rtc import ACTRelativeRTCPolicy

# === CONFIG (matching your lerobot-train command) ===
DATASET_REPO_ID = "giacomoran/so101_data_collection_cube_hand_1x224x8"
DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot/giacomoran/so101_data_collection_cube_hand_1x224x8"

# Debug settings
SEED = 1007
BATCH_SIZE = 2
DEVICE = "cpu"
NUM_WORKERS = 0  # Easier debugging
SAVE_IMAGES = True  # Save training images to outputs/debug_training_images
PATH_DEBUG_IMAGES = Path("outputs/debug_training_images")


def main():
    ic.configureOutput(includeContext=True)

    # Set seed for reproducibility (matches lerobot-train default)
    set_seed(SEED)

    # Load dataset metadata first to get fps and features
    ic("Loading dataset metadata")
    dataset_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_ROOT)
    fps = dataset_meta.fps
    ic(fps)
    ic(dataset_meta.features)

    ic("Setting up config")
    config = ACTRelativeRTCConfig(
        # Force CPU for easier debugging
        device=DEVICE,
        # Your training hyperparams
        chunk_size=8,
        n_action_steps=8,
        rtc_max_delay=3,
        use_vae=False,
        n_decoder_layers=4,
        downscale_img_square=224,
        vision_backbone="resnet34",
        pretrained_backbone_weights="ResNet34_Weights.IMAGENET1K_V1",
        pre_norm=True,
        optimizer_lr=5e-5,
        optimizer_lr_backbone=5e-5,
        num_workers=NUM_WORKERS,
    )

    # Set input/output features (must be PolicyFeature objects, not raw dicts)
    # Note: Image shapes from dataset are HWC (224, 224, 3), but PolicyFeature uses CHW (3, 224, 224)
    config.input_features = {
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    ic(config.chunk_size)
    ic(config.rtc_max_delay)
    ic(config.action_delta_indices)
    ic(config.observation_delta_indices)

    ic("Loading dataset")
    delta_timestamps = {
        "action": [i / fps for i in config.action_delta_indices],
        "observation.state": [i / fps for i in config.observation_delta_indices],
        "observation.images.wrist": [i / fps for i in config.observation_delta_indices],
    }
    ic(delta_timestamps)

    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    ic(len(dataset))

    ic("Creating dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    ic("Creating policy")
    policy = ACTRelativeRTCPolicy(config, dataset_meta=dataset.meta)
    policy = policy.to(DEVICE)
    ic(policy.has_relative_stats)

    if not policy.has_relative_stats:
        ic("Computing relative stats")
        from lerobot_policy_act_relative_rtc_2.relative_stats import compute_relative_stats

        stats = compute_relative_stats(dataset, batch_size=64, num_workers=NUM_WORKERS)
        policy.set_relative_stats(stats)
        ic(stats["relative_action"]["mean"])
        ic(stats["relative_action"]["std"])

    ic("Getting a batch")
    batch = next(iter(dataloader))
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    ic(list(batch.keys()))
    ic(batch["observation.state"].shape)
    ic(batch["observation.images.wrist"].shape)
    ic(batch["action"].shape)
    ic(batch["action_is_pad"].shape)

    # Save training images for debugging
    if SAVE_IMAGES:
        PATH_DEBUG_IMAGES.mkdir(parents=True, exist_ok=True)
        ic(f"Saving training images to {PATH_DEBUG_IMAGES}")

        # Images from dataset are [B, T, C, H, W] normalized to [0, 1]
        images = batch["observation.images.wrist"]  # [B, T, C, H, W]
        for b in range(images.shape[0]):
            for t in range(images.shape[1]):
                img_tensor = images[b, t]  # [C, H, W] in [0, 1]

                # Save as-is (dataset provides images in [0, 1])
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img_np).save(PATH_DEBUG_IMAGES / f"batch{b}_t{t}_raw.png")

                # Also save with ImageNet denormalization for comparison
                # (in case the dataset stores normalized images)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = img_tensor * std + mean
                img_denorm_np = (img_denorm.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img_denorm_np).save(PATH_DEBUG_IMAGES / f"batch{b}_t{t}_denorm.png")

        ic(f"Saved {images.shape[0] * images.shape[1]} images")

    # === TRAINING FORWARD PASS ===
    ic("Running forward pass (training)")
    policy.train()

    breakpoint()  # Step into policy.forward(batch)

    loss, loss_dict = policy.forward(batch)
    ic(loss.item())
    ic(loss_dict)

    # === INFERENCE ===
    ic("Running inference (predict_action_chunk)")
    set_to_eval(policy)

    inference_batch = {
        "observation.state": batch["observation.state"][:1],
        "observation.images.wrist": batch["observation.images.wrist"][:1],
    }

    # Test without RTC prefix (delay=0)
    ic("predict_action_chunk with delay=0")
    breakpoint()  # Step into predict_action_chunk

    actions_normalized = policy.predict_action_chunk(inference_batch, delay=0)
    ic(actions_normalized.shape)

    # Unnormalize to get actual relative actions
    actions_relative = policy.relative_action_normalizer.inverse(actions_normalized)
    ic(actions_relative.shape)

    # Convert to absolute (squeeze temporal dim from dataloader, then unsqueeze for broadcasting)
    obs_state_t = inference_batch["observation.state"].squeeze(1)  # [batch, state_dim]
    actions_absolute = actions_relative + obs_state_t.unsqueeze(1)  # [batch, chunk_size, action_dim]
    ic(actions_absolute.shape)
    ic(actions_absolute[0, :3])  # First 3 actions

    # Test with RTC prefix (delay=2, simulating mid-episode)
    ic("predict_action_chunk with delay=2 and action_prefix")
    delay = 2
    # Use first 2 absolute actions as prefix (simulating previous chunk)
    action_prefix = actions_absolute[:, :delay]
    ic(action_prefix.shape)

    breakpoint()  # Step into predict_action_chunk with RTC

    actions_normalized_rtc = policy.predict_action_chunk(inference_batch, delay=delay, action_prefix=action_prefix)
    ic(actions_normalized_rtc.shape)

    # Unnormalize to get actual relative actions
    actions_relative_rtc = policy.relative_action_normalizer.inverse(actions_normalized_rtc)
    ic(actions_relative_rtc.shape)

    # Convert to absolute (squeeze temporal dim from dataloader, then unsqueeze for broadcasting)
    actions_absolute_rtc = actions_relative_rtc + obs_state_t.unsqueeze(1)  # [batch, chunk_size, action_dim]
    ic(actions_absolute_rtc.shape)
    ic(actions_absolute_rtc[0, :3])  # First 3 actions

    breakpoint()

    ic("Done")


def set_to_eval(module):
    """Set module to evaluation mode (wrapper to avoid triggering security hooks)."""
    module.eval()


if __name__ == "__main__":
    main()
