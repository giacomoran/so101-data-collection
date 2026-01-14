#!/usr/bin/env python
"""Evaluate ACTRelativeRTC policy sensitivity to action prefix delay and translation.

This script tests how the policy responds to:
1. Different delay values (0 to rtc_max_delay) with ground truth action prefix
2. Translated/offset action prefixes at max delay

Usage:
    python -m so101_data_collection.eval.eval_rtc_sensitivity \
        --policy_path=outputs/some_rtc_policy/pretrained_model \
        --dataset_repo_id=giacomoran/cube_hand_guided \
        --episode_idx=0 \
        --n_observations=5 \
        --seed=42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging
from lerobot_policy_act_relative_rtc import ACTRelativeRTCPolicy


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalRTCSensitivityConfig:
    """Configuration for RTC sensitivity evaluation."""

    policy_path: str = ""
    dataset_repo_id: str = "giacomoran/cube_hand_guided"
    episode_idx: int = 0
    n_observations: int = 5
    seed: int = 42
    output_dir: str = "outputs/eval"
    n_translations: int = 5
    translation_max: float = 0.1
    video_backend: str = "pyav"
    device: str | None = None
    joint_idx: int = 0  # Which joint to plot

    def __post_init__(self):
        if not self.policy_path:
            raise ValueError("--policy_path must be provided")


# ============================================================================
# Dataset Loading
# ============================================================================


def load_dataset_for_episode(
    repo_id: str,
    episode_idx: int,
    rtc_max_delay: int,
    chunk_size: int,
    fps: int,
    video_backend: str,
    dataset_meta: LeRobotDatasetMetadata,
    obs_state_delta_frames: int,
) -> LeRobotDataset:
    """Load dataset with extended delta_timestamps to include past actions.

    Args:
        repo_id: HuggingFace dataset repo ID
        episode_idx: Episode index to load
        rtc_max_delay: Maximum RTC delay from policy config
        chunk_size: Chunk size from policy config
        fps: Dataset FPS
        video_backend: Video backend to use
        dataset_meta: Dataset metadata
        obs_state_delta_frames: Number of frames for observation delta

    Returns:
        LeRobotDataset with extended action delta_timestamps
    """
    # Build delta_timestamps with negative offsets for actions
    # This allows us to get past actions as action prefix
    delta_timestamps = {
        "action": [i / fps for i in range(-rtc_max_delay, chunk_size)],
        "observation.state": [-obs_state_delta_frames / fps, 0],
    }

    # Include image keys if present
    for key in dataset_meta.features:
        if key.startswith("observation.images."):
            delta_timestamps[key] = [-obs_state_delta_frames / fps, 0]

    dataset = LeRobotDataset(
        repo_id,
        episodes=[episode_idx],
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )

    return dataset


# ============================================================================
# Sampling
# ============================================================================


def sample_observation_indices(
    episode_from_idx: int,
    episode_to_idx: int,
    rtc_max_delay: int,
    chunk_size: int,
    n_samples: int,
    seed: int,
) -> list[int]:
    """Sample observation indices from valid range in episode.

    Valid range ensures:
    - Enough history for action prefix at max delay
    - Enough future for ground truth chunk

    Args:
        episode_from_idx: Episode start index in dataset
        episode_to_idx: Episode end index in dataset
        rtc_max_delay: Maximum RTC delay
        chunk_size: Action chunk size
        n_samples: Number of samples to draw
        seed: Random seed

    Returns:
        List of sampled observation indices (global dataset indices)
    """
    # Valid range: [from + rtc_max_delay, to - chunk_size)
    valid_from = episode_from_idx + rtc_max_delay
    valid_to = episode_to_idx - chunk_size

    if valid_to <= valid_from:
        raise ValueError(f"Episode too short for sampling: valid range [{valid_from}, {valid_to}) is empty")

    rng = np.random.RandomState(seed)
    indices_sampled = rng.choice(
        range(valid_from, valid_to),
        size=min(n_samples, valid_to - valid_from),
        replace=False,
    )

    return sorted(indices_sampled.tolist())


# ============================================================================
# Prefix Computation
# ============================================================================


def compute_prefix(
    sample: dict,
    policy: ACTRelativeRTCPolicy,
    rtc_max_delay: int,
    delay: int,
    device: torch.device,
    offset: float = 0.0,
) -> torch.Tensor | None:
    """Compute action prefix from episode data.

    The prefix consists of the FIRST `delay` actions from the ground truth chunk.
    This simulates having already predicted these actions in a previous inference step.

    Args:
        sample: Dataset sample with extended action delta_timestamps
        policy: Policy instance
        rtc_max_delay: Maximum RTC delay
        delay: Current delay value
        device: Device to use
        offset: Offset to add to prefix values (applied to relative, pre-normalization)
                Applied proportionally: action i gets offset * (i+1) / delay

    Returns:
        Normalized relative action prefix [1, delay, action_dim] or None if delay=0
    """
    if delay == 0:
        return None

    # Extract state at t (current timestep)
    obs_state_stacked = sample["observation.state"]
    state_t = obs_state_stacked[1]  # [state_dim]

    # Extract ground truth chunk starting at current timestep
    # sample["action"][:rtc_max_delay] are past actions (negative time offsets)
    # sample["action"][rtc_max_delay:] are future actions starting at t (ground truth chunk)
    actions_all = sample["action"]  # [rtc_max_delay + chunk_size, action_dim]
    gt_chunk = actions_all[rtc_max_delay:]  # [chunk_size, action_dim]

    # Prefix is the first `delay` actions from ground truth chunk
    prefix_absolute = gt_chunk[:delay]  # [delay, action_dim]

    # Convert to relative (relative to current state)
    prefix_relative = prefix_absolute - state_t  # [delay, action_dim]

    # Add offset if specified (proportionally spreading out)
    if offset != 0.0:
        # Create proportional offsets: action i gets offset * (i+1) / delay
        proportional_offsets = torch.linspace(1.0 / delay, 1.0, delay).unsqueeze(1).to(prefix_relative.device)
        prefix_relative = prefix_relative + offset * proportional_offsets

    # Normalize if policy has relative stats
    prefix_relative_tensor = prefix_relative.unsqueeze(0).to(device)  # [1, delay, action_dim]
    if policy.has_relative_stats:
        prefix_normalized = policy.relative_action_normalizer(prefix_relative_tensor)
    else:
        prefix_normalized = prefix_relative_tensor

    return prefix_normalized


# ============================================================================
# Inference
# ============================================================================


def predict_with_prefix(
    policy: ACTRelativeRTCPolicy,
    preprocessor,
    sample: dict,
    device: torch.device,
    delay: int,
    action_prefix: torch.Tensor | None,
) -> np.ndarray:
    """Predict action chunk with given prefix.

    Args:
        policy: Policy instance
        preprocessor: Preprocessor pipeline
        sample: Dataset sample
        device: Device to use
        delay: Delay value
        action_prefix: Normalized relative action prefix [1, delay, action_dim] or None

    Returns:
        Predicted absolute action chunk [chunk_size, action_dim]
    """
    # Extract state at t-1 and t
    obs_state_stacked = sample["observation.state"]
    state_t_minus_1 = obs_state_stacked[0]
    state_t = obs_state_stacked[1]
    delta_obs = state_t - state_t_minus_1

    # Build observation batch
    batch = {"observation.state": state_t.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[-1]  # Get image at t (last frame)
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
    with torch.no_grad():
        # Debug logging
        if action_prefix is not None:
            logging.debug(
                f"Prefix shape: {action_prefix.shape}, mean: {action_prefix.mean():.4f}, std: {action_prefix.std():.4f}"
            )

        relative_actions_normalized = policy.predict_action_chunk(
            batch,
            delay=delay,
            action_prefix=action_prefix,
        )

    # Unnormalize relative actions
    if policy.has_relative_stats:
        relative_actions = policy.relative_action_normalizer.inverse(relative_actions_normalized)
    else:
        relative_actions = relative_actions_normalized

    # Convert to absolute
    state_t_tensor = state_t.unsqueeze(0).to(device)
    absolute_actions = relative_actions + state_t_tensor.unsqueeze(1)

    return absolute_actions[0].cpu().numpy()


# ============================================================================
# Experiments
# ============================================================================


def run_delay_experiment(
    policy: ACTRelativeRTCPolicy,
    preprocessor,
    sample: dict,
    rtc_max_delay: int,
    device: torch.device,
) -> dict:
    """Run delay experiment for a single observation.

    Args:
        policy: Policy instance
        preprocessor: Preprocessor pipeline
        sample: Dataset sample
        rtc_max_delay: Maximum RTC delay
        device: Device to use

    Returns:
        Dictionary with delays and predictions:
        {
            "delays": [0, 1, 2, ...],
            "predictions": [pred_0, pred_1, ...],  # each is [chunk_size, action_dim]
            "prefixes": [None, prefix_1, ...],  # each is [delay, action_dim] in absolute coords, or None
        }
    """
    delays_list = list(range(rtc_max_delay + 1))
    predictions_list = []
    prefixes_list = []

    for delay in delays_list:
        # Compute prefix
        action_prefix = compute_prefix(sample, policy, rtc_max_delay, delay, device)

        # Get prefix for plotting (absolute coordinates to match ground truth)
        if delay > 0:
            actions_all = sample["action"]
            gt_chunk = actions_all[rtc_max_delay:]
            prefix_absolute = gt_chunk[:delay]
            prefixes_list.append(prefix_absolute.cpu().numpy())
        else:
            prefixes_list.append(None)

        # Predict
        pred = predict_with_prefix(policy, preprocessor, sample, device, delay, action_prefix)
        predictions_list.append(pred)

    return {
        "delays": delays_list,
        "predictions": predictions_list,
        "prefixes": prefixes_list,
    }


def run_translation_experiment(
    policy: ACTRelativeRTCPolicy,
    preprocessor,
    sample: dict,
    rtc_max_delay: int,
    n_translations: int,
    translation_max: float,
    device: torch.device,
) -> dict:
    """Run translation experiment for a single observation.

    Args:
        policy: Policy instance
        preprocessor: Preprocessor pipeline
        sample: Dataset sample
        rtc_max_delay: Maximum RTC delay (used as fixed delay)
        n_translations: Number of translation levels
        translation_max: Maximum translation offset
        device: Device to use

    Returns:
        Dictionary with offsets and predictions:
        {
            "offsets": [-max, ..., 0, ..., +max],
            "predictions": [pred_0, pred_1, ...],  # each is [chunk_size, action_dim]
            "prefixes": [prefix_0, prefix_1, ...],  # each is [delay, action_dim] in absolute coords
        }
    """
    offsets_list = np.linspace(-translation_max, translation_max, n_translations).tolist()
    predictions_list = []
    prefixes_list = []
    delay = rtc_max_delay

    for offset in offsets_list:
        # Compute prefix with offset
        action_prefix = compute_prefix(sample, policy, rtc_max_delay, delay, device, offset=offset)

        # Get prefix for plotting (absolute coordinates with offset applied proportionally)
        actions_all = sample["action"]
        gt_chunk = actions_all[rtc_max_delay:]
        prefix_absolute = gt_chunk[:delay].cpu().numpy()
        # Apply offset proportionally: action i gets offset * (i+1) / delay
        proportional_offsets = np.linspace(1.0 / delay, 1.0, delay).reshape(-1, 1)
        prefix_with_offset = prefix_absolute + offset * proportional_offsets
        prefixes_list.append(prefix_with_offset)

        # Predict
        pred = predict_with_prefix(policy, preprocessor, sample, device, delay, action_prefix)
        predictions_list.append(pred)

    return {
        "offsets": offsets_list,
        "predictions": predictions_list,
        "prefixes": prefixes_list,
    }


# ============================================================================
# Plotting
# ============================================================================


def create_plot(
    idx_samples: list[int],
    samples: list[dict],
    delay_results: list[dict],
    translation_results: list[dict],
    rtc_max_delay: int,
    joint_idx: int,
    chunk_size: int,
    fps: int,
    output_path: Path,
    use_rtc: bool,
) -> None:
    """Create and save sensitivity plot.

    Args:
        idx_samples: List of sample indices
        samples: List of dataset samples
        delay_results: List of delay experiment results
        translation_results: List of translation experiment results
        rtc_max_delay: Maximum RTC delay
        joint_idx: Which joint to plot
        chunk_size: Action chunk size
        fps: Dataset FPS
        output_path: Path to save plot
        use_rtc: Whether policy uses RTC
    """
    n_obs = len(idx_samples)
    fig, axes = plt.subplots(n_obs, 2, figsize=(16, 4 * n_obs))

    if n_obs == 1:
        axes = axes.reshape(1, -1)

    for idx_row, (idx_sample, sample, delay_res, trans_res) in enumerate(
        zip(idx_samples, samples, delay_results, translation_results)
    ):
        # Ground truth chunk (convert from relative to show alongside predictions)
        actions_all = sample["action"]
        gt_chunk = actions_all[rtc_max_delay:].cpu().numpy()  # [chunk_size, action_dim]
        gt_joint = gt_chunk[:, joint_idx]
        time_axis = np.arange(chunk_size) / fps

        # Current observation state
        obs_state_stacked = sample["observation.state"]
        state_t = obs_state_stacked[1].cpu().numpy()
        current_joint = state_t[joint_idx]

        # === Column 0: Delay Experiment ===
        ax = axes[idx_row, 0]

        # Plot current observation position
        ax.plot(0, current_joint, "ko", markersize=10, label="Current State", zorder=10)

        # Plot ground truth (dotted since it always matches the prefix)
        ax.plot(time_axis, gt_joint, "k:", linewidth=2, alpha=0.6, label="Ground Truth")

        # Plot predictions with colormap
        cmap = plt.cm.viridis
        for idx_delay, (delay, pred, prefix) in enumerate(
            zip(delay_res["delays"], delay_res["predictions"], delay_res["prefixes"])
        ):
            pred_joint = pred[:, joint_idx]
            color = cmap(idx_delay / rtc_max_delay)
            ax.plot(time_axis, pred_joint, color=color, alpha=0.6, linewidth=1.5, label=f"delay={delay}")

            # Plot prefix if available (should overlay exactly on ground truth)
            if prefix is not None:
                prefix_joint = prefix[:, joint_idx]
                prefix_time = np.arange(len(prefix)) / fps
                ax.plot(prefix_time, prefix_joint, "k--", alpha=0.8, linewidth=1.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Joint {joint_idx} (rad)")
        title_suffix = " (RTC disabled)" if not use_rtc else ""
        ax.set_title(f"Observation {idx_sample}: Delay Experiment{title_suffix}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # === Column 1: Translation Experiment ===
        ax = axes[idx_row, 1]

        # Plot current observation position
        ax.plot(0, current_joint, "ko", markersize=10, label="Current State", zorder=10)

        # Plot ground truth (dotted for consistency with delay experiment)
        ax.plot(time_axis, gt_joint, "k:", linewidth=2, alpha=0.6, label="Ground Truth")

        # Plot predictions with diverging colormap
        cmap = plt.cm.coolwarm
        n_trans = len(trans_res["offsets"])
        for idx_trans, (offset, pred, prefix) in enumerate(
            zip(trans_res["offsets"], trans_res["predictions"], trans_res["prefixes"])
        ):
            pred_joint = pred[:, joint_idx]
            color = cmap(idx_trans / (n_trans - 1))
            ax.plot(
                time_axis,
                pred_joint,
                color=color,
                alpha=0.6,
                linewidth=1.5,
                label=f"offset={offset:.3f}",
            )

            # Plot prefix (diverges proportionally from ground truth with offset)
            prefix_joint = prefix[:, joint_idx]
            prefix_time = np.arange(len(prefix)) / fps
            ax.plot(prefix_time, prefix_joint, color=color, linestyle="--", alpha=0.8, linewidth=1.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Joint {joint_idx} (rad)")
        ax.set_title(f"Observation {idx_sample}: Translation Experiment{title_suffix}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved plot to {output_path}")
    plt.show()


# ============================================================================
# Main
# ============================================================================


@draccus.wrap()
def main(cfg: EvalRTCSensitivityConfig):
    """Main entry point for RTC sensitivity evaluation."""
    init_logging()

    # Setup device
    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    logging.info(f"Using device: {device}")

    # Load policy
    logging.info(f"Loading policy from {cfg.policy_path}")
    policy = ACTRelativeRTCPolicy.from_pretrained(cfg.policy_path, device=device)
    policy.eval()

    # Use policy's actual device (may differ from requested due to auto-detection)
    device = policy.config.device

    use_rtc = policy.config.use_rtc
    rtc_max_delay = policy.config.rtc_max_delay

    if not use_rtc:
        logging.warning(
            "Policy has use_rtc=False. Experiments will run but predictions should be identical "
            "across all delay/offset values since prefix is not used."
        )
    chunk_size = policy.config.chunk_size
    obs_state_delta_frames = getattr(policy.config, "obs_state_delta_frames", 1)
    logging.info(
        f"Policy config: use_rtc={use_rtc}, rtc_max_delay={rtc_max_delay}, chunk_size={chunk_size}, "
        f"obs_state_delta_frames={obs_state_delta_frames}, device={device}"
    )

    # Create preprocessor
    preprocessor_overrides = {
        "device_processor": {"device": str(device)},
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Load dataset metadata
    logging.info(f"Loading dataset metadata from {cfg.dataset_repo_id}")
    dataset_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)
    fps = dataset_meta.fps

    # Get episode boundaries
    from_idx = dataset_meta.episodes["dataset_from_index"][cfg.episode_idx]
    to_idx = dataset_meta.episodes["dataset_to_index"][cfg.episode_idx]
    from_idx = int(from_idx.item() if hasattr(from_idx, "item") else from_idx)
    to_idx = int(to_idx.item() if hasattr(to_idx, "item") else to_idx)
    logging.info(f"Episode {cfg.episode_idx}: indices [{from_idx}, {to_idx})")

    # Load dataset
    logging.info("Loading dataset with extended delta_timestamps")
    dataset = load_dataset_for_episode(
        cfg.dataset_repo_id,
        cfg.episode_idx,
        rtc_max_delay,
        chunk_size,
        fps,
        cfg.video_backend,
        dataset_meta,
        obs_state_delta_frames,
    )

    # Sample observation indices
    logging.info(f"Sampling {cfg.n_observations} observations")
    idx_samples = sample_observation_indices(
        from_idx,
        to_idx,
        rtc_max_delay,
        chunk_size,
        cfg.n_observations,
        cfg.seed,
    )
    logging.info(f"Sampled indices: {idx_samples}")

    # Run experiments
    logging.info("Running experiments...")
    delay_results = []
    translation_results = []
    samples = []

    for idx_sample in idx_samples:
        logging.info(f"  Processing observation {idx_sample}")
        sample = dataset[idx_sample]
        samples.append(sample)

        # Delay experiment
        delay_res = run_delay_experiment(policy, preprocessor, sample, rtc_max_delay, device)
        delay_results.append(delay_res)

        # Translation experiment
        trans_res = run_translation_experiment(
            policy,
            preprocessor,
            sample,
            rtc_max_delay,
            cfg.n_translations,
            cfg.translation_max,
            device,
        )
        translation_results.append(trans_res)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plot
    output_path = (
        output_dir / f"rtc_sensitivity_ep{cfg.episode_idx}_j{cfg.joint_idx}_n{cfg.n_observations}_seed{cfg.seed}.png"
    )
    logging.info("Creating plot...")
    create_plot(
        idx_samples,
        samples,
        delay_results,
        translation_results,
        rtc_max_delay,
        cfg.joint_idx,
        chunk_size,
        fps,
        output_path,
        use_rtc,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
