#!/usr/bin/env python
"""Evaluate ACTRelativeRTC V2 policy sensitivity to action prefix delay and translation.

This script tests how the policy responds to:
1. Different delay values (0 to rtc_max_delay) with ground truth action prefix
2. Translated/offset action prefixes at max delay

V2 model key difference: For delay=d, the model predicts actions at t+d+1 to t+d+chunk_size
(shifted window), not the same window with prefix hints. This means each delay predicts
a different time window, and the ground truth must be extracted accordingly.

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
from lerobot_policy_act_relative_rtc_2 import ACTRelativeRTCPolicy


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
    translation_max: float = 0.3
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
) -> LeRobotDataset:
    """Load dataset with extended delta_timestamps for V2 model evaluation.

    V2 change: Actions start at index 1 (t+1), not 0. We load rtc_max_delay + chunk_size
    actions to cover all delay variations.

    Args:
        repo_id: HuggingFace dataset repo ID
        episode_idx: Episode index to load
        rtc_max_delay: Maximum RTC delay from policy config
        chunk_size: Chunk size from policy config
        fps: Dataset FPS
        video_backend: Video backend to use
        dataset_meta: Dataset metadata

    Returns:
        LeRobotDataset with extended action delta_timestamps
    """
    # Build delta_timestamps for V2 model
    # V2 model skips action[0] (always ~0 for relative actions) and uses indices [1, ...]
    # For delay=d, we need:
    #   - Prefix: actions at t+1, ..., t+d (indices 0 to d-1 in loaded data)
    #   - Ground truth: actions at t+d+1, ..., t+d+chunk_size (indices d to d+chunk_size-1)
    # So we need indices 1 to rtc_max_delay+chunk_size (total: rtc_max_delay+chunk_size actions)
    delta_timestamps = {
        "action": [i / fps for i in range(1, rtc_max_delay + chunk_size + 1)],
        "observation.state": [0],  # V2: single observation at t
    }

    # Include image keys if present
    for key in dataset_meta.features:
        if key.startswith("observation.images."):
            delta_timestamps[key] = [0]  # V2: single image at t

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
    - Enough future actions for all delays up to rtc_max_delay + chunk_size

    V2 change: No past history needed (removed negative delta_timestamps).

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
    # Valid range: [from, to - rtc_max_delay - chunk_size)
    # V2: No past history needed; need future actions up to t+rtc_max_delay+chunk_size
    valid_from = episode_from_idx
    valid_to = episode_to_idx - rtc_max_delay - chunk_size

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
    rtc_max_delay: int,
    delay: int,
    device: torch.device,
    offset: float = 0.0,
) -> torch.Tensor | None:
    """Compute action prefix from episode data.

    V2 change: Actions now start at t+1 (index 0), not t-rtc_max_delay.

    The prefix consists of the FIRST `delay` actions (t+1 to t+delay).
    This simulates having already predicted these actions in a previous inference step.

    Args:
        sample: Dataset sample with action delta_timestamps starting at 1/fps
        rtc_max_delay: Maximum RTC delay (unused in V2, kept for API compatibility)
        delay: Current delay value
        device: Device to use
        offset: Offset to add to prefix values (applied to relative, pre-normalization)
                Applied proportionally: action i gets offset * (i+1) / delay

    Returns:
        Absolute action prefix [1, delay, action_dim] or None if delay=0
        (predict_action_chunk converts to relative and normalizes internally)
    """
    if delay == 0:
        return None

    # V2: sample["action"][0] is action at t+1 (first action after observation)
    # sample["action"][i] is action at t+1+i
    # For delay=d, prefix = actions at t+1, ..., t+d = sample["action"][0:d]
    actions_all = sample["action"]  # [rtc_max_delay + chunk_size, action_dim]

    # Prefix is the first `delay` actions (t+1 to t+delay)
    prefix_absolute = actions_all[:delay]  # [delay, action_dim]

    # Add offset if specified (proportionally spreading out)
    # Since predict_action_chunk computes relative = absolute - state_t internally,
    # adding offset to absolute results in the same effect as adding to relative.
    if offset != 0.0:
        # Create proportional offsets: action i gets offset * (i+1) / delay
        proportional_offsets = torch.linspace(1.0 / delay, 1.0, delay).unsqueeze(1).to(prefix_absolute.device)
        prefix_absolute = prefix_absolute + offset * proportional_offsets

    # Return absolute prefix (predict_action_chunk handles conversion to relative + normalization)
    return prefix_absolute.unsqueeze(0).to(device)  # [1, delay, action_dim]


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

    V2 note: For delay=d, the model predicts actions at t+d+1 to t+d+chunk_size.

    Args:
        policy: Policy instance
        preprocessor: Preprocessor pipeline
        sample: Dataset sample
        device: Device to use
        delay: Delay value
        action_prefix: Absolute action prefix [1, delay, action_dim] or None
                       (V2 predict_action_chunk converts to relative internally)

    Returns:
        Predicted absolute action chunk [chunk_size, action_dim]
    """
    # Extract state at t (V2: single timestamp, squeeze temporal dim)
    state_t = sample["observation.state"].squeeze(0)  # [state_dim]

    # Build observation batch with current state
    batch = {"observation.state": state_t.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[-1]  # Get image at t (last frame)
            batch[key] = img.unsqueeze(0).to(device)

    # Apply preprocessor
    batch = preprocessor(batch)

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
    chunk_size: int,
    device: torch.device,
) -> dict:
    """Run delay experiment for a single observation.

    V2 change: Each delay predicts a shifted window. Ground truth for delay=d
    is actions at t+d+1 to t+d+chunk_size.

    Args:
        policy: Policy instance
        preprocessor: Preprocessor pipeline
        sample: Dataset sample
        rtc_max_delay: Maximum RTC delay
        chunk_size: Action chunk size
        device: Device to use

    Returns:
        Dictionary with delays and predictions:
        {
            "delays": [0, 1, 2, ...],
            "predictions": [pred_0, pred_1, ...],  # each is [chunk_size, action_dim]
            "prefixes": [None, prefix_1, ...],  # each is [delay, action_dim] in absolute coords, or None
            "ground_truths": [gt_0, gt_1, ...],  # each is [chunk_size, action_dim], shifted per delay
        }
    """
    delays_list = list(range(rtc_max_delay + 1))
    predictions_list = []
    prefixes_list = []
    ground_truths_list = []

    actions_all = sample["action"]  # [rtc_max_delay + chunk_size, action_dim]

    for delay in delays_list:
        # Compute prefix
        action_prefix = compute_prefix(sample, rtc_max_delay, delay, device)

        # Get prefix for plotting (absolute coordinates)
        # V2: prefix = actions_all[0:delay] = actions at t+1, ..., t+delay
        if delay > 0:
            prefix_absolute = actions_all[:delay]
            prefixes_list.append(prefix_absolute.cpu().numpy())
        else:
            prefixes_list.append(None)

        # Get ground truth for this delay
        # V2: For delay=d, ground truth = actions_all[d:d+chunk_size] = actions at t+d+1, ..., t+d+chunk_size
        gt_for_delay = actions_all[delay : delay + chunk_size]
        ground_truths_list.append(gt_for_delay.cpu().numpy())

        # Predict
        pred = predict_with_prefix(policy, preprocessor, sample, device, delay, action_prefix)
        predictions_list.append(pred)

    return {
        "delays": delays_list,
        "predictions": predictions_list,
        "prefixes": prefixes_list,
        "ground_truths": ground_truths_list,
    }


def run_translation_experiment(
    policy: ACTRelativeRTCPolicy,
    preprocessor,
    sample: dict,
    rtc_max_delay: int,
    chunk_size: int,
    n_translations: int,
    translation_max: float,
    device: torch.device,
) -> dict:
    """Run translation experiment for a single observation.

    V2 change: Ground truth for max_delay is actions at t+rtc_max_delay+1 to t+rtc_max_delay+chunk_size.

    Args:
        policy: Policy instance
        preprocessor: Preprocessor pipeline
        sample: Dataset sample
        rtc_max_delay: Maximum RTC delay (used as fixed delay)
        chunk_size: Action chunk size
        n_translations: Number of translation levels
        translation_max: Maximum translation offset
        device: Device to use

    Returns:
        Dictionary with offsets and predictions:
        {
            "offsets": [-max, ..., 0, ..., +max],
            "predictions": [pred_0, pred_1, ...],  # each is [chunk_size, action_dim]
            "prefixes": [prefix_0, prefix_1, ...],  # each is [delay, action_dim] in absolute coords
            "ground_truth": [chunk_size, action_dim],  # single ground truth for fixed delay=rtc_max_delay
        }
    """
    offsets_list = np.linspace(-translation_max, translation_max, n_translations).tolist()
    predictions_list = []
    prefixes_list = []
    delay = rtc_max_delay

    actions_all = sample["action"]  # [rtc_max_delay + chunk_size, action_dim]

    # V2: Ground truth for delay=rtc_max_delay is actions_all[rtc_max_delay:rtc_max_delay+chunk_size]
    ground_truth = actions_all[delay : delay + chunk_size].cpu().numpy()

    for offset in offsets_list:
        # Compute prefix with offset
        action_prefix = compute_prefix(sample, rtc_max_delay, delay, device, offset=offset)

        # Get prefix for plotting (absolute coordinates with offset applied proportionally)
        # V2: prefix_absolute = actions_all[0:delay] = actions at t+1, ..., t+delay
        prefix_absolute = actions_all[:delay].cpu().numpy()
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
        "ground_truth": ground_truth,
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
) -> None:
    """Create and save sensitivity plot.

    V2 changes:
    - Time axis starts at 1/fps (actions start at t+1)
    - For delay=d, predictions are for actions at t+d+1 to t+d+chunk_size
    - Each delay has its own ground truth
    - Prefix for delay=d covers actions at t+1 to t+d

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
    """
    n_obs = len(idx_samples)
    fig, axes = plt.subplots(n_obs, 2, figsize=(16, 4 * n_obs))

    if n_obs == 1:
        axes = axes.reshape(1, -1)

    for idx_row, (idx_sample, sample, delay_res, trans_res) in enumerate(
        zip(idx_samples, samples, delay_results, translation_results)
    ):
        # Current observation state (V2: single timestamp, squeeze temporal dim)
        state_t = sample["observation.state"]
        if state_t.dim() > 1:
            state_t = state_t.squeeze(0)
        state_t = state_t.cpu().numpy()
        current_joint = state_t[joint_idx]

        # === Column 0: Delay Experiment ===
        ax = axes[idx_row, 0]

        # Plot current observation position at t=0
        ax.plot(0, current_joint, "ko", markersize=10, label="Current State (t)", zorder=10)

        # Plot predictions with colormap, each with its own time axis and ground truth
        cmap = plt.cm.viridis
        for idx_delay, (delay, pred, prefix, gt) in enumerate(
            zip(delay_res["delays"], delay_res["predictions"], delay_res["prefixes"], delay_res["ground_truths"])
        ):
            color = cmap(idx_delay / rtc_max_delay)

            # V2: For delay=d, predictions are for t+d+1 to t+d+chunk_size
            # Time axis for predictions: (d+1)/fps to (d+chunk_size)/fps
            pred_time_axis = np.arange(chunk_size) / fps + (delay + 1) / fps
            pred_joint = pred[:, joint_idx]
            ax.plot(pred_time_axis, pred_joint, color=color, alpha=0.7, linewidth=1.5, label=f"delay={delay}")

            # Plot ground truth for this delay (same time axis as prediction)
            gt_joint = gt[:, joint_idx]
            ax.plot(pred_time_axis, gt_joint, color=color, linestyle=":", alpha=0.5, linewidth=1.5)

            # Plot prefix if available
            # V2: Prefix for delay=d covers actions at t+1 to t+d, i.e., times 1/fps to d/fps
            if prefix is not None:
                prefix_joint = prefix[:, joint_idx]
                prefix_time = np.arange(1, delay + 1) / fps
                ax.plot(prefix_time, prefix_joint, color=color, linestyle="--", alpha=0.6, linewidth=1.5)

        ax.set_xlabel("Time from observation (s)")
        ax.set_ylabel(f"Joint {joint_idx} (rad)")
        ax.set_title(f"Obs {idx_sample}: Delay Experiment (solid=pred, dotted=GT)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

        # === Column 1: Translation Experiment ===
        ax = axes[idx_row, 1]

        # Plot current observation position at t=0
        ax.plot(0, current_joint, "ko", markersize=10, label="Current State (t)", zorder=10)

        # V2: For translation experiment, delay=rtc_max_delay
        # Predictions are for t+rtc_max_delay+1 to t+rtc_max_delay+chunk_size
        pred_time_axis = np.arange(chunk_size) / fps + (rtc_max_delay + 1) / fps

        # Plot ground truth (fixed for all translations at delay=rtc_max_delay)
        gt_joint = trans_res["ground_truth"][:, joint_idx]
        ax.plot(pred_time_axis, gt_joint, "k:", linewidth=2, alpha=0.6, label="Ground Truth")

        # Plot predictions with diverging colormap
        cmap = plt.cm.coolwarm
        n_trans = len(trans_res["offsets"])
        for idx_trans, (offset, pred, prefix) in enumerate(
            zip(trans_res["offsets"], trans_res["predictions"], trans_res["prefixes"])
        ):
            pred_joint = pred[:, joint_idx]
            color = cmap(idx_trans / (n_trans - 1))
            ax.plot(
                pred_time_axis,
                pred_joint,
                color=color,
                alpha=0.7,
                linewidth=1.5,
                label=f"offset={offset:.3f}",
            )

            # Plot prefix (times 1/fps to rtc_max_delay/fps)
            prefix_joint = prefix[:, joint_idx]
            prefix_time = np.arange(1, rtc_max_delay + 1) / fps
            ax.plot(prefix_time, prefix_joint, color=color, linestyle="--", alpha=0.6, linewidth=1.5)

        ax.set_xlabel("Time from observation (s)")
        ax.set_ylabel(f"Joint {joint_idx} (rad)")
        ax.set_title(f"Obs {idx_sample}: Translation Experiment (delay={rtc_max_delay})")
        ax.legend(loc="best", fontsize=7)
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

    rtc_max_delay = policy.config.rtc_max_delay
    chunk_size = policy.config.chunk_size
    logging.info(f"Policy config: rtc_max_delay={rtc_max_delay}, chunk_size={chunk_size}, device={device}")

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
        delay_res = run_delay_experiment(policy, preprocessor, sample, rtc_max_delay, chunk_size, device)
        delay_results.append(delay_res)

        # Translation experiment
        trans_res = run_translation_experiment(
            policy,
            preprocessor,
            sample,
            rtc_max_delay,
            chunk_size,
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
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
