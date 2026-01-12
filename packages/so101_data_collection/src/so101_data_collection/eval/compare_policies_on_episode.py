#!/usr/bin/env python
"""Compare multiple policies' action predictions against ground truth on a dataset episode.

This script helps verify policy behavior before deployment on hardware by comparing
predicted action chunks against ground truth from the training data.

Supports different policy types (ACT, ACT Relative RTC, etc.) and action representations
(absolute vs relative joint positions).

Usage:
    python src/zxtra/compare_policies_on_episode.py \
        --policy_paths=[outputs/policy_0/pretrained_model,outputs/policy_1/pretrained_model] \
        --dataset_repo_id=giacomoran/so101_data_collection_cube_hand_guided \
        --dataset_episode_idx=[0,1]

    # Can also use HuggingFace repo IDs:
    python src/zxtra/compare_policies_on_episode.py \
        --policy_paths=[user/repo_name] \
        --dataset_repo_id=giacomoran/so101_data_collection_cube_hand_guided \
        --dataset_episode_idx=[0]
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import init_logging

# ============================================================================
# Protocol for policies with different action representations
# ============================================================================


@runtime_checkable
class RelativeActionPolicy(Protocol):
    """Protocol for policies that output relative actions (need conversion to absolute)."""

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict relative action chunk."""
        ...


def is_relative_action_policy(policy: PreTrainedPolicy) -> bool:
    """Check if a policy uses relative action representation."""
    # Check by policy type name or config class
    policy_type = getattr(policy.config, "type", None) or policy.name
    return policy_type in ("act_relative_rtc",)


# ============================================================================
# Policy loading utilities
# ============================================================================


def load_policy_from_path(
    pretrained_name_or_path: str,
    device: str,
) -> tuple[PreTrainedPolicy, str]:
    """Load any policy from a local path or HuggingFace repo ID.

    Uses PreTrainedPolicy.from_pretrained() which handles both local paths
    and HuggingFace repos. For custom policies (act_relative_rtc), uses
    direct import and from_pretrained().

    Returns:
        Tuple of (policy, policy_type_name)
    """
    path = Path(pretrained_name_or_path)

    # Check if it's a local path or HuggingFace repo ID
    # If it has multiple path segments, it's likely a local path
    is_local_path = len(path.parts) > 1 or path.exists()

    # Determine policy type from config
    if is_local_path:
        # Local path - check if it's a directory with config.json
        if path.is_dir() and (path / "config.json").exists():
            config_path = path / "config.json"
        elif path.parent.is_dir() and (path.parent / "config.json").exists():
            # Path might be to model.safetensors, use parent
            config_path = path.parent / "config.json"
            path = path.parent
        else:
            config_path = path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        policy_type = config_dict.get("type", None)
        if policy_type is None:
            raise ValueError(f"Config at {config_path} missing 'type' field")

        pretrained_path = str(path)
    else:
        # HuggingFace repo ID - download config to determine policy type
        from huggingface_hub import hf_hub_download

        # Download config.json to determine policy type
        config_file = hf_hub_download(
            repo_id=pretrained_name_or_path,
            filename="config.json",
        )

        with open(config_file) as f:
            config_dict = json.load(f)

        policy_type = config_dict.get("type", None)
        if policy_type is None:
            raise ValueError(
                f"Config from {pretrained_name_or_path} missing 'type' field"
            )

        pretrained_path = pretrained_name_or_path

    # Load policy using from_pretrained()
    if policy_type == "act_relative_rtc":
        # For act_relative_rtc, use direct import (not registered in factory)
        from lerobot_policy_act_relative_rtc import ACTRelativeRTCPolicy

        policy = ACTRelativeRTCPolicy.from_pretrained(
            pretrained_path,
            device=device,
        )
    else:
        # Use lerobot's factory for standard policies
        from lerobot.policies.factory import get_policy_class

        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(
            pretrained_path,
            device=device,
        )

    policy.eval()
    source = "local path" if is_local_path else "HuggingFace"
    logging.info(f"Loaded {policy_type} policy from {source}: {pretrained_path}")
    return policy, policy_type


# ============================================================================
# Inference utilities
# ============================================================================


def get_delta_timestamps_for_policy(
    policy: PreTrainedPolicy,
    fps: float,
    chunk_size: int,
) -> dict[str, list[float]]:
    """Get delta_timestamps based on policy type."""
    action_timestamps = [i / fps for i in range(chunk_size)]

    if is_relative_action_policy(policy):
        # ACT Relative RTC needs obs[t-1] and obs[t] for delta computation
        return {
            "observation.state": [-1 / fps, 0],
            "observation.images.wrist": [-1 / fps, 0],
            "observation.images.top": [-1 / fps, 0],
            "action": action_timestamps,
        }
    else:
        # Standard policies just need current observation
        return {
            "observation.state": [0],
            "observation.images.wrist": [0],
            "observation.images.top": [0],
            "action": action_timestamps,
        }


def predict_action_chunk_absolute(
    policy: PreTrainedPolicy,
    sample: dict[str, Any],
    device: str,
    action_prefix: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Predict an action chunk and convert to absolute positions.

    Handles both absolute (ACT) and relative (ACT Relative RTC) action representations.

    Args:
        policy: The policy to use for prediction
        sample: Dataset sample containing observations
        device: Device to run inference on
        action_prefix: Optional action prefix for RTC, shape [delay, action_dim]

    Returns:
        pred_actions_absolute: [chunk_size, action_dim] absolute joint positions
        inference_time: time taken for inference in seconds
    """
    if is_relative_action_policy(policy):
        # ACT Relative RTC: input is delta observation, output is relative actions
        obs_state_stacked = sample["observation.state"]  # [2, state_dim]
        obs_t_minus_1 = obs_state_stacked[0]  # [state_dim]
        obs_t = obs_state_stacked[1]  # [state_dim]

        # Compute delta observation
        delta_obs = obs_t - obs_t_minus_1  # [state_dim]

        # Prepare batch
        batch = {
            "observation.state": delta_obs.unsqueeze(0).to(device),  # [1, state_dim]
        }

        # Add images (use current frame, index 1)
        for key in sample:
            if key.startswith("observation.images."):
                img = sample[key]  # [2, C, H, W]
                batch[key] = img[1:2].to(device)  # [1, C, H, W] - current frame

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            if action_prefix is not None and policy.config.use_rtc:
                action_prefix_tensor = (
                    torch.from_numpy(action_prefix).unsqueeze(0).to(device)
                )  # [1, delay, action_dim]
                relative_actions = policy.predict_action_chunk(
                    batch,
                    delay=action_prefix.shape[0],
                    action_prefix=action_prefix_tensor,
                )  # [1, chunk_size, action_dim]
            else:
                relative_actions = policy.predict_action_chunk(
                    batch
                )  # [1, chunk_size, action_dim]
        inference_time = time.perf_counter() - start_time

        # Convert relative to absolute: action_abs = action_rel + obs[t]
        obs_t_expanded = obs_t.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, state_dim]
        absolute_actions = (
            relative_actions + obs_t_expanded
        )  # [1, chunk_size, action_dim]
        pred_actions = absolute_actions[0].cpu().numpy()  # [chunk_size, action_dim]

    else:
        # Standard ACT: input is absolute observation, output is absolute actions
        batch = {}

        if "observation.state" in sample:
            obs_state = sample["observation.state"]
            # Handle both stacked [2, state_dim] and single [state_dim] formats
            if obs_state.ndim == 2:  # [2, state_dim] stacked
                obs_state = obs_state[-1]  # Take current frame [state_dim]
            batch["observation.state"] = obs_state.unsqueeze(0).to(device)

        for key in sample:
            if key.startswith("observation.images."):
                img = sample[key]
                # Handle both stacked [2, C, H, W] and single [C, H, W] formats
                if img.ndim == 4:  # [2, C, H, W] stacked
                    img = img[-1]  # Take current frame [C, H, W]
                batch[key] = img.unsqueeze(0).to(device)  # [1, C, H, W]

        start_time = time.perf_counter()
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)  # [1, chunk_size, action_dim]
        inference_time = time.perf_counter() - start_time

        pred_actions = actions[0].cpu().numpy()  # [chunk_size, action_dim]

    return pred_actions, inference_time


# ============================================================================
# Main comparison logic
# ============================================================================


@dataclass
class ComparePoliciesConfig:
    """Configuration for policy comparison."""

    policy_paths: list[str] = field(default_factory=list)
    dataset_repo_id: str = "giacomoran/so101_data_collection_cube_hand_guided"
    dataset_episode_idx: list[int] = field(default_factory=lambda: [0])
    device: str | None = None
    plot_interval: float = 1.0
    rtc_delay: int = 0

    def __post_init__(self):
        if not self.policy_paths:
            raise ValueError(
                "At least one policy path must be provided via --policy_paths"
            )
        if not self.dataset_episode_idx:
            raise ValueError(
                "At least one episode index must be provided via --dataset_episode_idx"
            )


def run_comparison_for_episode(
    policies: list[PreTrainedPolicy],
    policy_names: list[str],
    cfg: ComparePoliciesConfig,
    episode_idx: int,
    ds_meta: LeRobotDatasetMetadata,
):
    """Run comparison for a single episode."""
    fps = ds_meta.fps
    chunk_size = policies[0].config.chunk_size

    from_idx = ds_meta.episodes["dataset_from_index"][episode_idx]
    to_idx = ds_meta.episodes["dataset_to_index"][episode_idx]
    from_idx = int(from_idx.item() if hasattr(from_idx, "item") else from_idx)
    to_idx = int(to_idx.item() if hasattr(to_idx, "item") else to_idx)
    num_frames = to_idx - from_idx

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Episode {episode_idx}: {num_frames} frames at {fps} fps")
    logging.info(f"  Dataset indices: [{from_idx}, {to_idx})")
    logging.info(f"  RTC delay: {cfg.rtc_delay}")
    logging.info(f"{'=' * 60}")

    # Get action dimension
    action_dim = policies[0].config.action_feature.shape[0]
    dim_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "gripper",
    ][:action_dim]

    # We need to load the dataset with the most permissive delta_timestamps
    # (the one that includes obs[t-1] if any policy needs it)
    needs_prev_obs = any(is_relative_action_policy(p) for p in policies)

    # Build delta_timestamps based on what features exist in the dataset
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}

    if needs_prev_obs:
        state_ts = [-1 / fps, 0]
        image_ts = [-1 / fps, 0]
    else:
        state_ts = [0]
        image_ts = [0]

    # Add observation.state if present
    if "observation.state" in ds_meta.features:
        delta_timestamps["observation.state"] = state_ts

    # Add image features if present
    for key in ds_meta.features:
        if key.startswith("observation.images."):
            delta_timestamps[key] = image_ts

    # Create dataset for this specific episode
    # Note: LeRobotDataset with episodes=[episode_idx] should re-index frames starting from 0
    # However, there seems to be a bug where episode 1+ returns the same frame repeatedly
    # So we'll use global indices as a workaround
    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        episodes=None,  # Load all episodes, we'll index manually
        delta_timestamps=delta_timestamps,
    )

    # Storage for each policy
    all_inference_times = {name: [] for name in policy_names}
    all_chunk_errors = {name: [] for name in policy_names}
    per_dim_errors = {name: [[] for _ in range(action_dim)] for name in policy_names}

    # Store chunks at each second for each policy
    second_chunks_predicted = {name: [] for name in policy_names}

    # Collect ground truth actions using global indices
    logging.info("Collecting ground truth actions...")
    gt_actions_list = []

    for local_idx in range(num_frames):
        # Use global dataset index (from_idx + local_idx)
        global_idx = from_idx + local_idx
        try:
            sample = dataset[global_idx]
        except (IndexError, KeyError) as e:
            logging.error(
                f"Failed to access dataset[{global_idx}]: {e}. "
                f"Dataset length: {len(dataset)}, episode range: [{from_idx}, {to_idx})"
            )
            raise

        gt_chunk = sample["action"]
        if isinstance(gt_chunk, torch.Tensor):
            gt_chunk = gt_chunk.numpy()

        # Extract the first action from the chunk (action at current timestep)
        # The action chunk from delta_timestamps contains [action[t], action[t+1], ...]
        # So gt_chunk[0] is the action at the current timestep
        if len(gt_chunk) == 0:
            logging.warning(f"Frame {local_idx}: Empty action chunk!")
            # Use zeros as fallback
            action_dim = policies[0].config.action_feature.shape[0]
            gt_actions_list.append(np.zeros(action_dim))
        else:
            gt_actions_list.append(gt_chunk[0])

    gt_actions_episode = np.array(gt_actions_list)  # [num_frames, action_dim]
    logging.info(f"Ground truth shape: {gt_actions_episode.shape}")

    # Run inference for all policies
    logging.info("Running inference on all observations...")

    for policy, name in zip(policies, policy_names):
        policy.reset()

    for local_idx in range(num_frames):
        # Use global dataset index (from_idx + local_idx)
        global_idx = from_idx + local_idx
        sample = dataset[global_idx]

        gt_chunk = sample["action"]
        if isinstance(gt_chunk, torch.Tensor):
            gt_chunk = gt_chunk.numpy()

        for idx_policy, (policy, name) in enumerate(zip(policies, policy_names)):
            action_prefix = None
            if cfg.rtc_delay > 0 and len(gt_chunk) > 0 and policy.config.use_rtc:
                prefix_len = min(cfg.rtc_delay, len(gt_chunk))
                action_prefix = gt_chunk[:prefix_len]

            pred_chunk, inference_time = predict_action_chunk_absolute(
                policy, sample, cfg.device, action_prefix=action_prefix
            )
            all_inference_times[name].append(inference_time)

            # Compute error
            valid_len = min(len(pred_chunk), len(gt_chunk))
            error = np.abs(pred_chunk[:valid_len] - gt_chunk[:valid_len])
            mean_error = error.mean()
            all_chunk_errors[name].append(mean_error)

            for dim in range(action_dim):
                per_dim_errors[name][dim].append(error[:, dim].mean())

            # Store chunks at each interval
            if local_idx % int(fps * cfg.plot_interval) == 0:
                second_chunks_predicted[name].append((local_idx, pred_chunk.copy()))

        if local_idx % 100 == 0:
            errors_str = ", ".join(
                f"{name.split('/')[-1][:10]}: {all_chunk_errors[name][-1]:.4f}"
                for name in policy_names
            )
            logging.info(f"  Frame {local_idx}/{num_frames} - {errors_str}")

    # === Summary Statistics ===
    logging.info("\n" + "=" * 60)
    logging.info(f"COMPARISON RESULTS - Episode {episode_idx}")
    logging.info("=" * 60)

    for name in policy_names:
        logging.info(f"\n📊 {name}")
        logging.info("-" * 40)

        # Inference latency
        inference_times = np.array(all_inference_times[name])
        logging.info(
            f"  Latency: {inference_times.mean() * 1000:.2f}ms ± {inference_times.std() * 1000:.2f}ms"
        )
        logging.info(
            f"  Effective Hz: {1.0 / inference_times.mean():.1f} Hz (target: {fps} Hz)"
        )

        # Action errors
        all_errors = np.array(all_chunk_errors[name])
        logging.info(f"  Mean Error: {all_errors.mean():.4f} ± {all_errors.std():.4f}")

        # Per-dimension error
        for dim in range(action_dim):
            dim_err = np.array(per_dim_errors[name][dim])
            dim_name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
            logging.info(f"    {dim_name:15s}: {dim_err.mean():.4f}")

    # === Create Plots ===
    logging.info(f"\n📊 Generating plots for episode {episode_idx}...")

    # Color palette for policies
    colors = plt.cm.tab10(np.linspace(0, 1, len(policy_names)))

    # Create 4x2 grid plot
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()

    fig.suptitle(
        f"Action Comparison: {len(policy_names)} Policies vs Ground Truth\n"
        f"Dataset: {cfg.dataset_repo_id}, Episode: {episode_idx}",
        fontsize=14,
    )

    time_axis = np.arange(num_frames) / fps

    # First subplots: action comparison for all joints
    for dim in range(action_dim):
        ax = axes[dim]
        dim_name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"

        # Ground truth
        ax.plot(
            time_axis,
            gt_actions_episode[:, dim],
            "gray",
            label="Ground Truth",
            linewidth=1.0,
            alpha=0.3,
        )

        # Plot each chunk separately for each policy
        for i, name in enumerate(policy_names):
            short_name = name.split("/")[-1].split(" ")[0][:15]
            for start_frame, pred_chunk in second_chunks_predicted[name]:
                chunk_time = np.arange(start_frame, start_frame + len(pred_chunk)) / fps
                ax.plot(
                    chunk_time,
                    pred_chunk[:, dim],
                    "--",
                    color=colors[i],
                    linewidth=1,
                    alpha=0.5,
                )
            # Add legend entry
            ax.plot([], [], "--", color=colors[i], label=short_name)

        ax.set_title(dim_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Action", fontsize=8)
        ax.legend(fontsize=7, loc="best", ncol=min(len(policy_names) + 1, 3))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Next subplot: error over time
    ax = axes[action_dim]
    for i, name in enumerate(policy_names):
        short_name = name.split("/")[-1].split(" ")[0][:15]
        errors = np.array(all_chunk_errors[name])
        ax.plot(
            time_axis,
            errors,
            color=colors[i],
            linewidth=1,
            alpha=0.7,
            label=f"{short_name}",
        )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Mean L1 Error", fontsize=8)
    ax.set_title("Error Over Time", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Last subplot: per-dimension error comparison (bar chart)
    ax = axes[action_dim + 1]
    x = np.arange(action_dim)
    width = 0.8 / len(policy_names)

    for i, name in enumerate(policy_names):
        short_name = name.split("/")[-1].split(" ")[0][:15]
        dim_means = [
            np.array(per_dim_errors[name][d]).mean() for d in range(action_dim)
        ]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            dim_means,
            width,
            label=short_name,
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Joint", fontsize=8)
    ax.set_ylabel("Mean L1 Error", fontsize=8)
    ax.set_title("Per-Dimension Error", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    logging.info(f"✅ Episode {episode_idx} complete! Showing plots...")
    plt.show()


def run_comparison(cfg: ComparePoliciesConfig):
    """Run policy comparison."""
    init_logging()

    # Auto-select device
    if cfg.device is None:
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"
    logging.info(f"Using device: {cfg.device}")

    # Load all policies
    policies = []
    policy_names = []
    for path_str in cfg.policy_paths:
        policy, policy_type = load_policy_from_path(path_str, cfg.device)
        policies.append(policy)
        # Create a clean label, removing "pretrained_model" if present
        path = Path(path_str)
        # Handle both local paths and HuggingFace repo IDs
        if "/" in path_str and not path.exists() and not path.parent.exists():
            # Likely a HuggingFace repo ID (no local path exists)
            label = path_str.split("/")[-1]
        else:
            # Local path - clean up the name
            parts = path.parts
            # Remove "pretrained_model" if it's the last part
            if parts and parts[-1] == "pretrained_model":
                parts = parts[:-1]
            # Use last 2 parts for label if available
            if len(parts) >= 2:
                label = "/".join(parts[-2:])
            elif len(parts) == 1:
                label = parts[0]
            else:
                label = str(path)
        policy_names.append(f"{label} ({policy_type})")

    if not policies:
        logging.error("No policies loaded!")
        return

    # Validate rtc_delay against policy configs
    for policy, policy_type in zip(policies, policy_names):
        if hasattr(policy.config, "use_rtc") and policy.config.use_rtc:
            max_delay = policy.config.rtc_max_delay
            if cfg.rtc_delay > max_delay:
                raise ValueError(
                    f"rtc_delay ({cfg.rtc_delay}) exceeds rtc_max_delay ({max_delay}) "
                    f"for policy {policy_type}"
                )

    # Get chunk_size from first policy (assume consistent across policies)
    chunk_size = policies[0].config.chunk_size
    logging.info(f"Using chunk_size: {chunk_size}")

    # Load dataset metadata
    ds_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)

    # Run comparison for each episode
    for episode_idx in cfg.dataset_episode_idx:
        run_comparison_for_episode(policies, policy_names, cfg, episode_idx, ds_meta)


@draccus.wrap()
def main(cfg: ComparePoliciesConfig):
    run_comparison(cfg)


if __name__ == "__main__":
    main()
