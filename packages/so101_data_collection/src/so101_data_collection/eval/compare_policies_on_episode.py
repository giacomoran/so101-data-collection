#!/usr/bin/env python
"""Compare multiple policies' action predictions against ground truth on a dataset episode.

This script helps verify policy behavior before deployment on hardware by comparing
predicted action chunks against ground truth from the training data.

Supports different policy types (ACT, ACT Relative RTC, etc.) with separate
inference paths for clarity.

Usage:
    python src/so101_data_collection/eval/compare_policies_on_episode.py \
        --policy_paths=[outputs/policy_0/pretrained_model,outputs/policy_1/pretrained_model] \
        --dataset_repo_id=giacomoran/so101_data_collection_cube_hand_guided \
        --dataset_episode_idx=[0,1]

    # Can also use HuggingFace repo IDs:
    python src/so101_data_collection/eval/compare_policies_on_episode.py \
        --policy_paths=[user/repo_name] \
        --dataset_repo_id=giacomoran/so101_data_collection_cube_hand_guided \
        --dataset_episode_idx=[0]
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.utils import init_logging

# ============================================================================
# Configuration
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
    video_backend: str = "pyav"

    def __post_init__(self):
        if not self.policy_paths:
            raise ValueError("At least one policy path must be provided via --policy_paths")
        if not self.dataset_episode_idx:
            raise ValueError("At least one episode index must be provided via --dataset_episode_idx")


# ============================================================================
# Policy Loading
# ============================================================================


def load_policy_and_preprocessor(
    pretrained_name_or_path: str,
    device: str,
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, str]:
    """Load policy and its preprocessor from a local path or HuggingFace repo ID.

    Returns:
        Tuple of (policy, preprocessor, policy_type_name)
    """
    path = Path(pretrained_name_or_path)

    is_local_path = len(path.parts) > 1 or path.exists()

    # Determine policy type from config
    if is_local_path:
        if path.is_dir() and (path / "config.json").exists():
            config_path = path / "config.json"
        elif path.parent.is_dir() and (path.parent / "config.json").exists():
            config_path = path.parent / "config.json"
            path = path.parent
        else:
            config_path = path / "config.json"

        with open(config_path) as f:
            config_dict = json.load(f)

        policy_type = config_dict.get("type", None)
        if policy_type is None:
            raise ValueError(f"Config at {config_path} missing 'type' field")

        pretrained_path = str(path)
    else:
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(
            repo_id=pretrained_name_or_path,
            filename="config.json",
        )

        with open(config_file) as f:
            config_dict = json.load(f)

        policy_type = config_dict.get("type", None)
        if policy_type is None:
            raise ValueError(f"Config from {pretrained_name_or_path} missing 'type' field")

        pretrained_path = pretrained_name_or_path

    # Load policy
    if policy_type == "act_relative_rtc":
        from lerobot_policy_act_relative_rtc import ACTRelativeRTCPolicy

        policy = ACTRelativeRTCPolicy.from_pretrained(
            pretrained_path,
            device=device,
        )
    elif policy_type == "act_relative_rtc_2":
        from lerobot_policy_act_relative_rtc_2 import ACTRelativeRTCPolicy

        policy = ACTRelativeRTCPolicy.from_pretrained(
            pretrained_path,
            device=device,
        )
    else:
        from lerobot.policies.factory import get_policy_class

        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(
            pretrained_path,
            device=device,
        )

    policy.eval()

    # Create preprocessor
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    source = "local path" if is_local_path else "HuggingFace"
    logging.info(f"Loaded {policy_type} policy from {source}: {pretrained_path}")
    return policy, preprocessor, policy_type


def is_relative_action_policy(policy: PreTrainedPolicy) -> bool:
    """Check if a policy uses relative action representation (V1 or V2)."""
    policy_type = getattr(policy.config, "type", None) or policy.name
    return policy_type in ("act_relative_rtc", "act_relative_rtc_2")


def is_relative_action_policy_v2(policy: PreTrainedPolicy) -> bool:
    """Check if a policy is the V2 relative action policy (no delta_obs input)."""
    policy_type = getattr(policy.config, "type", None) or policy.name
    return policy_type == "act_relative_rtc_2"


# ============================================================================
# Inference - Relative Policy (ACTRelativeRTC)
# ============================================================================


def predict_chunk_relative(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    sample: dict,
    device: str,
    robot_type: str,
    rtc_delay: int = 0,
) -> tuple[np.ndarray, float]:
    """Predict absolute action chunk from ACTRelativeRTC policy.

    Returns:
        (absolute_actions, inference_time_seconds)
    """
    # Extract state at t-1 and t from sample
    obs_state_stacked = sample["observation.state"]
    state_t_minus_1 = obs_state_stacked[0]
    state_t = obs_state_stacked[1]
    delta_obs = state_t - state_t_minus_1

    # Build observation batch (PyTorch tensors, images at t only)
    batch = {"observation.state": state_t.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[1]
            batch[key] = img.unsqueeze(0).to(device)

    # Apply preprocessor
    batch = preprocessor(batch)

    # Normalize delta observation
    delta_obs_tensor = delta_obs.unsqueeze(0).to(device)
    if policy.has_relative_stats:
        delta_obs_normalized = policy.delta_obs_normalizer(delta_obs_tensor)
    else:
        delta_obs_normalized = delta_obs_tensor

    # Update batch with delta observation
    batch["observation.state"] = delta_obs_normalized

    # Run inference
    start_time = time.perf_counter()
    with torch.no_grad():
        if rtc_delay > 0 and policy.config.use_rtc:
            gt_actions = sample["action"].cpu().numpy()
            prefix_len = min(rtc_delay, len(gt_actions))
            # Get absolute action prefix from ground truth
            action_prefix_abs = torch.from_numpy(gt_actions[:prefix_len]).unsqueeze(0).to(device)
            # Convert to relative: action_prefix - state_t
            state_t_tensor = state_t.unsqueeze(0).to(device)
            action_prefix_relative = action_prefix_abs - state_t_tensor.unsqueeze(1)
            # Normalize if has_relative_stats
            if policy.has_relative_stats:
                action_prefix_relative = policy.relative_action_normalizer(action_prefix_relative)
            relative_actions = policy.predict_action_chunk(
                batch, delay=prefix_len, action_prefix=action_prefix_relative
            )
        else:
            relative_actions = policy.predict_action_chunk(batch)
    inference_time = time.perf_counter() - start_time

    # Unnormalize relative actions
    if policy.has_relative_stats:
        relative_actions = policy.relative_action_normalizer.inverse(relative_actions)

    # Convert to absolute
    state_t_tensor = state_t.unsqueeze(0).to(device)
    absolute_actions = relative_actions + state_t_tensor

    return absolute_actions[0].cpu().numpy(), inference_time


# ============================================================================
# Inference - Relative Policy V2 (ACTRelativeRTC2 - no delta_obs input)
# ============================================================================


def predict_chunk_relative_v2(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    sample: dict,
    device: str,
    rtc_delay: int = 0,
) -> tuple[np.ndarray, float]:
    """Predict absolute action chunk from ACTRelativeRTC2 policy (V2).

    V2 differences from V1:
    - Uses raw observation state as input (not delta_obs)
    - No delta_obs_normalizer
    - predict_action_chunk expects absolute action_prefix (converts internally)

    Returns:
        (absolute_actions, inference_time_seconds)
    """
    # Extract state at t from sample (V2 only needs current state, not delta)
    obs_state_stacked = sample["observation.state"]
    if obs_state_stacked.dim() == 2:
        # [2, state_dim] -> take current state at t (index 1)
        state_t = obs_state_stacked[1]
    else:
        state_t = obs_state_stacked

    # Build observation batch (PyTorch tensors, images at t only)
    batch = {"observation.state": state_t.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[1]  # Take image at t
            batch[key] = img.unsqueeze(0).to(device)

    # Apply preprocessor
    batch = preprocessor(batch)

    # Run inference
    # Note: V2 always has RTC enabled (no use_rtc flag, just rtc_max_delay)
    start_time = time.perf_counter()
    with torch.no_grad():
        if rtc_delay > 0:
            gt_actions = sample["action"].cpu().numpy()
            prefix_len = min(rtc_delay, len(gt_actions))
            # Get absolute action prefix from ground truth
            # V2: predict_action_chunk expects absolute action_prefix (converts internally)
            action_prefix_abs = torch.from_numpy(gt_actions[:prefix_len]).unsqueeze(0).to(device)
            relative_actions_normalized = policy.predict_action_chunk(
                batch, delay=prefix_len, action_prefix=action_prefix_abs
            )
        else:
            relative_actions_normalized = policy.predict_action_chunk(batch)
    inference_time = time.perf_counter() - start_time

    # Unnormalize relative actions
    relative_actions = policy.relative_action_normalizer.inverse(relative_actions_normalized)

    # Convert to absolute
    state_t_tensor = state_t.unsqueeze(0).to(device)
    absolute_actions = relative_actions + state_t_tensor.unsqueeze(1)

    return absolute_actions[0].cpu().numpy(), inference_time


# ============================================================================
# Inference - Absolute Policy (standard ACT)
# ============================================================================


def predict_chunk_absolute(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    sample: dict,
    device: str,
) -> tuple[np.ndarray, float]:
    """Predict absolute action chunk from standard ACT policy.

    Returns:
        (absolute_actions, inference_time_seconds)
    """
    # Extract state at t
    obs_state = sample["observation.state"]
    if obs_state.ndim == 2:
        obs_state = obs_state[-1]

    # Build batch directly
    batch = {"observation.state": obs_state.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[-1]
            batch[key] = img.unsqueeze(0).to(device)

    # Apply preprocessor
    batch = preprocessor(batch)

    # Run inference
    start_time = time.perf_counter()
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    inference_time = time.perf_counter() - start_time

    return actions[0].cpu().numpy(), inference_time


# ============================================================================
# Episode Comparison
# ============================================================================


def run_comparison_for_episode(
    policies: list[PreTrainedPolicy],
    preprocessors: list[PolicyProcessorPipeline],
    policy_names: list[str],
    cfg: ComparePoliciesConfig,
    episode_idx: int,
    dataset_meta: LeRobotDatasetMetadata,
    robot_type: str,
) -> dict:
    """Run comparison for a single episode.

    Returns:
        Dictionary containing collected data for plotting:
        {
            "episode_idx": int,
            "time_axis": np.ndarray,
            "gt_actions": np.ndarray,
            "policy_predictions": {
                policy_name: {
                    "chunks": [(start_frame, pred_chunk), ...],
                    "errors_over_time": np.ndarray,
                    "per_dim_errors": list[np.ndarray],
                    "inference_times": np.ndarray,
                },
                ...
            },
            "dim_names": list[str],
        }
    """
    fps = dataset_meta.fps
    chunk_size = policies[0].config.chunk_size

    from_idx = dataset_meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset_meta.episodes["dataset_to_index"][episode_idx]
    from_idx = int(from_idx.item() if hasattr(from_idx, "item") else from_idx)
    to_idx = int(to_idx.item() if hasattr(to_idx, "item") else to_idx)
    num_frames = to_idx - from_idx

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Episode {episode_idx}: {num_frames} frames at {fps} fps")
    logging.info(f"  Dataset indices: [{from_idx}, {to_idx})")
    logging.info(f"  RTC delay: {cfg.rtc_delay}")
    logging.info(f"{'=' * 60}")

    # Build delta_timestamps (always include t-1 for state and images)
    delta_timestamps = {
        "action": [i / fps for i in range(chunk_size)],
        "observation.state": [-1 / fps, 0],
    }
    for key in dataset_meta.features:
        if key.startswith("observation.images."):
            delta_timestamps[key] = [-1 / fps, 0]

    # Load dataset for this episode
    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        episodes=[episode_idx],
        delta_timestamps=delta_timestamps,
        video_backend=cfg.video_backend,
    )

    # Get action dimension and joint names
    action_dim = policies[0].config.action_feature.shape[0]
    dim_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "gripper",
    ][:action_dim]

    # Initialize storage for each policy
    policy_predictions = {}
    for name in policy_names:
        policy_predictions[name] = {
            "chunks": [],
            "errors_over_time": [],
            "per_dim_errors": [[] for _ in range(action_dim)],
            "inference_times": [],
        }

    # Collect ground truth actions
    # Note: dataset is loaded with episodes=[episode_idx], so indices are 0 to num_frames-1
    logging.info("Collecting ground truth actions...")
    gt_actions_list = []
    for local_idx in range(num_frames):
        sample = dataset[local_idx]
        gt_chunk = sample["action"].cpu().numpy()
        gt_actions_list.append(gt_chunk[0])

    gt_actions = np.array(gt_actions_list)
    time_axis = np.arange(num_frames) / fps
    logging.info(f"Ground truth shape: {gt_actions.shape}")

    # Run inference for all policies
    logging.info("Running inference on all observations...")

    for policy in policies:
        policy.reset()

    for local_idx in range(num_frames):
        sample = dataset[local_idx]
        gt_chunk = sample["action"].cpu().numpy()

        for idx_policy, (policy, name, preprocessor) in enumerate(zip(policies, policy_names, preprocessors)):
            # Call appropriate inference function based on policy type
            if is_relative_action_policy_v2(policy):
                # V2: no delta_obs input, uses raw observation state
                pred_chunk, inference_time = predict_chunk_relative_v2(
                    policy,
                    preprocessor,
                    sample,
                    cfg.device,
                    rtc_delay=cfg.rtc_delay,
                )
            elif is_relative_action_policy(policy):
                # V1: uses delta_obs as input
                pred_chunk, inference_time = predict_chunk_relative(
                    policy,
                    preprocessor,
                    sample,
                    cfg.device,
                    robot_type,
                    rtc_delay=cfg.rtc_delay,
                )
            else:
                pred_chunk, inference_time = predict_chunk_absolute(
                    policy,
                    preprocessor,
                    sample,
                    cfg.device,
                )

            policy_predictions[name]["inference_times"].append(inference_time)

            # Compute error
            valid_len = min(len(pred_chunk), len(gt_chunk))
            error = np.abs(pred_chunk[:valid_len] - gt_chunk[:valid_len])
            policy_predictions[name]["errors_over_time"].append(error.mean())

            for dim in range(action_dim):
                policy_predictions[name]["per_dim_errors"][dim].append(error[:, dim].mean())

            # Store chunks at each interval
            if local_idx % int(fps * cfg.plot_interval) == 0:
                policy_predictions[name]["chunks"].append((local_idx, pred_chunk.copy()))

        if local_idx % 100 == 0:
            errors_str = ", ".join(
                f"{name.split('/')[-1][:10]}: {policy_predictions[name]['errors_over_time'][-1]:.4f}"
                for name in policy_names
            )
            logging.info(f"  Frame {local_idx}/{num_frames} - {errors_str}")

    # Log summary statistics
    logging.info("\n" + "=" * 60)
    logging.info(f"COMPARISON RESULTS - Episode {episode_idx}")
    logging.info("=" * 60)

    for name in policy_names:
        logging.info(f"\n📊 {name}")
        logging.info("-" * 40)

        inference_times = np.array(policy_predictions[name]["inference_times"])
        logging.info(f"  Latency: {inference_times.mean() * 1000:.2f}ms ± {inference_times.std() * 1000:.2f}ms")
        logging.info(f"  Effective Hz: {1.0 / inference_times.mean():.1f} Hz (target: {fps} Hz)")

        all_errors = np.array(policy_predictions[name]["errors_over_time"])
        logging.info(f"  Mean Error: {all_errors.mean():.4f} ± {all_errors.std():.4f}")

        for dim in range(action_dim):
            dim_err = np.array(policy_predictions[name]["per_dim_errors"][dim])
            dim_name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
            logging.info(f"    {dim_name:15s}: {dim_err.mean():.4f}")

    return {
        "episode_idx": episode_idx,
        "time_axis": time_axis,
        "gt_actions": gt_actions,
        "policy_predictions": policy_predictions,
        "dim_names": dim_names,
    }


# ============================================================================
# Plotting
# ============================================================================


def create_comparison_plots(
    comparison_data: dict,
    dataset_repo_id: str,
    dataset_meta: LeRobotDatasetMetadata,
):
    """Create and display comparison plots from collected data."""
    episode_idx = comparison_data["episode_idx"]
    time_axis = comparison_data["time_axis"]
    gt_actions = comparison_data["gt_actions"]
    policy_predictions = comparison_data["policy_predictions"]
    dim_names = comparison_data["dim_names"]
    policy_names = list(policy_predictions.keys())

    action_dim = gt_actions.shape[1]

    logging.info(f"\n📊 Generating plots for episode {episode_idx}...")

    colors = plt.cm.tab10(np.linspace(0, 1, len(policy_names)))

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()

    fig.suptitle(
        f"Action Comparison: {len(policy_names)} Policies vs Ground Truth\n"
        f"Dataset: {dataset_repo_id}, Episode: {episode_idx}",
        fontsize=14,
    )

    # Per-joint action comparison
    for dim in range(action_dim):
        ax = axes[dim]
        dim_name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"

        ax.plot(
            time_axis,
            gt_actions[:, dim],
            "gray",
            label="Ground Truth",
            linewidth=1.0,
            alpha=0.3,
        )

        for i, name in enumerate(policy_names):
            short_name = name.split(" (")[0]  # Remove policy type suffix
            for start_frame, pred_chunk in policy_predictions[name]["chunks"]:
                chunk_time = np.arange(start_frame, start_frame + len(pred_chunk)) / dataset_meta.fps
                ax.plot(
                    chunk_time,
                    pred_chunk[:, dim],
                    "--",
                    color=colors[i],
                    linewidth=1,
                    alpha=0.5,
                )
            ax.plot([], [], "--", color=colors[i], label=short_name)

        ax.set_title(dim_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Action", fontsize=8)
        ax.legend(fontsize=7, loc="best", ncol=min(len(policy_names) + 1, 3))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Error over time
    ax = axes[action_dim]
    for i, name in enumerate(policy_names):
        short_name = name.split(" (")[0]  # Remove policy type suffix
        errors = np.array(policy_predictions[name]["errors_over_time"])
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

    # Per-dimension error comparison
    ax = axes[action_dim + 1]
    x = np.arange(action_dim)
    width = 0.8 / len(policy_names)

    for i, name in enumerate(policy_names):
        short_name = name.split(" (")[0]  # Remove policy type suffix
        dim_means = [np.array(policy_predictions[name]["per_dim_errors"][d]).mean() for d in range(action_dim)]
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
    plt.show()


# ============================================================================
# Main Entry Point
# ============================================================================


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

    # Load all policies with preprocessors
    policies = []
    preprocessors = []
    policy_names = []
    for path_str in cfg.policy_paths:
        policy, preprocessor, policy_type = load_policy_and_preprocessor(path_str, cfg.device)
        policies.append(policy)
        preprocessors.append(preprocessor)

        path = Path(path_str)
        if "/" in path_str and not path.exists() and not path.parent.exists():
            # HuggingFace repo ID
            label = path_str.split("/")[-1]
        else:
            # Local path - extract meaningful label
            parts = list(path.parts)
            # Remove common suffixes from end
            if parts and parts[-1] == "pretrained_model":
                parts = parts[:-1]
            # Check for checkpoint pattern: .../checkpoints/NNNNNN
            if len(parts) >= 2 and parts[-2] == "checkpoints":
                checkpoint_num = parts[-1]
                parts = parts[:-2]  # Remove both 'checkpoints' and the number
                # Include checkpoint number in label
                if parts:
                    label = f"{parts[-1]}/{checkpoint_num}"
                else:
                    label = checkpoint_num
            elif len(parts) >= 2:
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
                    f"rtc_delay ({cfg.rtc_delay}) exceeds rtc_max_delay ({max_delay}) for policy {policy_type}"
                )

    # Load dataset metadata
    dataset_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)

    # Get robot type for inference
    robot_type = "so101_follower"
    logging.info(f"Using robot_type: {robot_type}")

    # Run comparison for each episode
    for episode_idx in cfg.dataset_episode_idx:
        comparison_data = run_comparison_for_episode(
            policies,
            preprocessors,
            policy_names,
            cfg,
            episode_idx,
            dataset_meta,
            robot_type,
        )
        create_comparison_plots(comparison_data, cfg.dataset_repo_id, dataset_meta)


@draccus.wrap()
def main(cfg: ComparePoliciesConfig):
    run_comparison(cfg)


if __name__ == "__main__":
    main()
