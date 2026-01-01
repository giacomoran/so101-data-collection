#!/usr/bin/env python
"""Validation script for overfitted ACT-UMI policy.

This script compares the policy's predicted action chunks against ground truth
actions from the training episode to validate the model before deployment.

Usage:
    python src/zxtra/validate_policy_overfit.py \
        --checkpoint_path=outputs/train_act_umi_overfit/checkpoints/002000/pretrained_model \
        --repo_id=giacomoran/so101_data_collection_cube_hand_guided \
        --episode_idx=0
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import init_logging
from safetensors.torch import load_model as load_model_as_safetensor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from act_umi import ACTUMIConfig, ACTUMIPolicy


@dataclass
class ValidatePolicyConfig:
    """Configuration for policy validation."""

    checkpoint_path: Path = Path(
        "outputs/train_act_umi_overfit/checkpoints/002000/pretrained_model"
    )
    repo_id: str = "giacomoran/so101_data_collection_cube_hand_guided"
    episode_idx: int = 0
    device: str | None = None
    output_dir: Path = Path("outputs/train_act_umi_overfit/checkpoints")


def load_policy_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> ACTUMIPolicy:
    """Load ACTUMIPolicy from checkpoint."""
    config_path = checkpoint_path / "config.json"
    model_path = checkpoint_path / SAFETENSORS_SINGLE_FILE

    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict.pop("type", None)

    # Convert feature dictionaries to PolicyFeature objects
    for features_key in ["input_features", "output_features"]:
        if features_key in config_dict:
            features = {}
            for key, ft_dict in config_dict[features_key].items():
                feature_type_map = {
                    "VISUAL": FeatureType.VISUAL,
                    "STATE": FeatureType.STATE,
                    "ACTION": FeatureType.ACTION,
                    "ENV": FeatureType.ENV,
                }
                features[key] = PolicyFeature(
                    type=feature_type_map[ft_dict["type"]],
                    shape=tuple(ft_dict["shape"]),
                )
            config_dict[features_key] = features

    config = ACTUMIConfig(**config_dict)
    config.device = device

    policy = ACTUMIPolicy(config)
    load_model_as_safetensor(policy, str(model_path))
    policy = policy.to(device)
    policy.eval()

    logging.info(f"Loaded policy from {checkpoint_path}")
    return policy


def load_episode_data(repo_id: str, episode_idx: int, chunk_size: int):
    """Load dataset for the specified episode."""
    ds_meta = LeRobotDatasetMetadata(repo_id)
    fps = ds_meta.fps

    from_idx = ds_meta.episodes["dataset_from_index"][episode_idx]
    to_idx = ds_meta.episodes["dataset_to_index"][episode_idx]
    from_idx = int(from_idx.item() if hasattr(from_idx, 'item') else from_idx)
    to_idx = int(to_idx.item() if hasattr(to_idx, 'item') else to_idx)
    num_frames = to_idx - from_idx

    logging.info(f"Episode {episode_idx}: {num_frames} frames at {fps} fps")

    # Dataset with delta timestamps for UMI policy (obs[t-1] and obs[t])
    delta_timestamps = {
        "observation.state": [-1 / fps, 0],
        "observation.images.wrist": [-1 / fps, 0],
        "observation.images.top": [-1 / fps, 0],
        "action": [i / fps for i in range(chunk_size)],
    }

    dataset = LeRobotDataset(
        repo_id,
        episodes=[episode_idx],
        delta_timestamps=delta_timestamps,
    )

    return dataset, num_frames, fps


def get_action_chunk_absolute(
    policy: ACTUMIPolicy,
    sample: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, float]:
    """Get a full action chunk from the policy, converted to absolute positions.

    Returns:
        pred_actions_absolute: [chunk_size, action_dim] absolute joint positions
        inference_time: time taken for inference in seconds
    """
    # Extract stacked observations [obs[t-1], obs[t]]
    obs_state_stacked = sample["observation.state"]  # [2, state_dim]
    obs_t_minus_1 = obs_state_stacked[0]  # [state_dim]
    obs_t = obs_state_stacked[1]  # [state_dim]

    # Compute delta observation for the policy input
    delta_obs = obs_t - obs_t_minus_1  # [state_dim]

    # Prepare batch
    batch = {
        "observation.state": delta_obs.unsqueeze(0).to(device),  # [1, state_dim]
    }

    # Add images (use current frame, index 1)
    for key in ["observation.images.wrist", "observation.images.top"]:
        img = sample[key]  # [2, C, H, W]
        batch[key] = img[1:2].to(device)  # [1, C, H, W] - current frame

    # Run inference
    start_time = time.perf_counter()
    with torch.no_grad():
        # predict_action_chunk returns RELATIVE actions
        relative_actions = policy.predict_action_chunk(batch)  # [1, chunk_size, action_dim]
    inference_time = time.perf_counter() - start_time

    # Debug: Check relative actions
    relative_actions_np = relative_actions[0].cpu().numpy()  # [chunk_size, action_dim]
    relative_std = relative_actions_np.std(axis=0)
    relative_mean = relative_actions_np.mean(axis=0)

    # Convert relative to absolute: action_abs = action_rel + obs[t]
    # relative_actions: [1, chunk_size, action_dim]
    # obs_t: [state_dim]
    obs_t_expanded = obs_t.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, state_dim]
    absolute_actions = relative_actions + obs_t_expanded  # [1, chunk_size, action_dim]

    pred_actions_absolute = absolute_actions[0].cpu().numpy()  # [chunk_size, action_dim]

    return pred_actions_absolute, inference_time, relative_mean, relative_std


def run_validation(cfg: ValidatePolicyConfig):
    """Main validation function."""
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

    # Load policy
    policy = load_policy_from_checkpoint(cfg.checkpoint_path, cfg.device)
    chunk_size = policy.config.chunk_size
    logging.info(f"Policy chunk size: {chunk_size}")

    # Load dataset
    dataset, num_frames, fps = load_episode_data(cfg.repo_id, cfg.episode_idx, chunk_size)
    logging.info(f"Dataset FPS: {fps}, frames: {num_frames}")

    action_dim = policy.config.action_feature.shape[0]
    dim_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "gripper"]

    # Storage
    all_inference_times = []
    all_chunk_errors = []  # Mean error per chunk
    per_dim_errors = [[] for _ in range(action_dim)]

    # For plotting: chunks predicted at each second (every fps frames)
    second_chunks_predicted = []  # List of (start_frame, predicted_chunk)

    # Debug: Track relative action statistics
    relative_action_stats = []

    # Collect ground truth actions (individual actions, not chunks)
    logging.info("Collecting ground truth actions...")
    gt_actions_list = []
    for local_idx in range(num_frames):
        sample = dataset[local_idx]
        # Ground truth is the action chunk starting at this frame
        gt_chunk = sample["action"]  # [chunk_size, action_dim]
        if isinstance(gt_chunk, torch.Tensor):
            gt_chunk = gt_chunk.numpy()
        # We just need the first action of each chunk = action at this timestep
        gt_actions_list.append(gt_chunk[0])
    gt_actions_episode = np.array(gt_actions_list)  # [num_frames, action_dim]
    logging.info(f"Ground truth shape: {gt_actions_episode.shape}")

    # Run inference for ALL observations (for stats)
    logging.info("Running inference on all observations...")
    policy.reset()

    for local_idx in range(num_frames):
        sample = dataset[local_idx]

        # Get predicted action chunk (absolute positions)
        pred_chunk, inference_time, rel_mean, rel_std = get_action_chunk_absolute(policy, sample, cfg.device)
        all_inference_times.append(inference_time)

        # Store relative action stats for first few frames
        if local_idx < 5:
            relative_action_stats.append({
                'frame': local_idx,
                'mean': rel_mean,
                'std': rel_std,
            })

        # Get ground truth chunk for comparison
        gt_chunk = sample["action"]
        if isinstance(gt_chunk, torch.Tensor):
            gt_chunk = gt_chunk.numpy()

        # Compute error (handle end of episode where gt might be padded)
        valid_len = min(len(pred_chunk), len(gt_chunk))
        error = np.abs(pred_chunk[:valid_len] - gt_chunk[:valid_len])
        mean_error = error.mean()
        all_chunk_errors.append(mean_error)

        for dim in range(action_dim):
            per_dim_errors[dim].append(error[:, dim].mean())

        # Store chunks at each second (0s, 1s, 2s, ...)
        if local_idx % fps == 0:
            second_chunks_predicted.append((local_idx, pred_chunk.copy()))
            # Diagnostic: check variance within chunk
            chunk_std = pred_chunk.std(axis=0)
            logging.info(f"  Frame {local_idx}: chunk std per dim: {chunk_std}")

        if local_idx % 100 == 0:
            logging.info(f"  Frame {local_idx}/{num_frames}, error: {mean_error:.4f}")

    # === Summary Statistics ===
    logging.info("\n" + "=" * 60)
    logging.info("VALIDATION RESULTS")
    logging.info("=" * 60)

    # Inference latency
    inference_times = np.array(all_inference_times)
    logging.info("\n📊 INFERENCE LATENCY STATS:")
    logging.info(f"  Mean:   {inference_times.mean() * 1000:.2f} ms")
    logging.info(f"  Std:    {inference_times.std() * 1000:.2f} ms")
    logging.info(f"  Min:    {inference_times.min() * 1000:.2f} ms")
    logging.info(f"  Max:    {inference_times.max() * 1000:.2f} ms")
    logging.info(f"  Median: {np.median(inference_times) * 1000:.2f} ms")
    logging.info(f"  P95:    {np.percentile(inference_times, 95) * 1000:.2f} ms")
    logging.info(f"  P99:    {np.percentile(inference_times, 99) * 1000:.2f} ms")
    effective_hz = 1.0 / inference_times.mean()
    logging.info(f"  Effective Hz: {effective_hz:.1f} Hz (target: {fps} Hz)")

    # Action errors
    all_errors = np.array(all_chunk_errors)
    logging.info("\n📊 ACTION ERROR STATS (L1, averaged over chunk):")
    logging.info(f"  Mean:   {all_errors.mean():.4f}")
    logging.info(f"  Std:    {all_errors.std():.4f}")
    logging.info(f"  Max:    {all_errors.max():.4f}")
    logging.info(f"  Min:    {all_errors.min():.4f}")

    logging.info("\n📊 PER-DIMENSION ERROR:")
    for dim in range(action_dim):
        dim_err = np.array(per_dim_errors[dim])
        name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
        logging.info(f"  {name:15s}: mean={dim_err.mean():.4f}, max={dim_err.max():.4f}")

    # Debug: Relative action statistics
    logging.info("\n📊 RELATIVE ACTION STATISTICS (first 5 frames):")
    logging.info("  This shows what relative actions the model is predicting.")
    logging.info("  If these are all near zero, the model learned to 'stay still'.")
    for stat in relative_action_stats:
        logging.info(f"  Frame {stat['frame']}:")
        for dim in range(action_dim):
            name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
            logging.info(f"    {name:15s}: mean={stat['mean'][dim]:.6f}, std={stat['std'][dim]:.6f}")

    # === Create Plots ===
    logging.info("\n📊 Creating plots...")

    # Diagnostic: Check if chunks are constant
    logging.info("\n📊 CHUNK VARIANCE ANALYSIS:")
    for start_frame, pred_chunk in second_chunks_predicted[:5]:  # First 5 chunks
        chunk_var = pred_chunk.var(axis=0)
        chunk_std = pred_chunk.std(axis=0)
        logging.info(f"  Chunk at frame {start_frame}:")
        for dim in range(action_dim):
            name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
            logging.info(f"    {name}: std={chunk_std[dim]:.4f}, range=[{pred_chunk[:, dim].min():.2f}, {pred_chunk[:, dim].max():.2f}]")

    # Build concatenated predicted trajectory from second-by-second chunks
    # Each chunk at second s covers frames [s*fps, (s+1)*fps)
    concatenated_pred = np.full((num_frames, action_dim), np.nan)

    for start_frame, pred_chunk in second_chunks_predicted:
        end_frame = min(start_frame + chunk_size, num_frames)
        chunk_len = end_frame - start_frame
        concatenated_pred[start_frame:end_frame] = pred_chunk[:chunk_len]

    # Plot: one subplot per joint
    fig, axes = plt.subplots(action_dim, 1, figsize=(14, 2.5 * action_dim), squeeze=False)
    axes = axes[:, 0]

    fig.suptitle("Action Comparison: Concatenated Predicted Chunks vs Ground Truth Episode",
                 fontsize=14, y=1.01)

    time_axis = np.arange(num_frames) / fps

    for dim in range(action_dim):
        ax = axes[dim]
        name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"

        # Ground truth (full episode)
        ax.plot(time_axis, gt_actions_episode[:, dim], 'b-',
                label='Ground Truth', linewidth=2, alpha=0.8)

        # Predicted (concatenated chunks)
        ax.plot(time_axis, concatenated_pred[:, dim], 'r--',
                label='Predicted (concatenated chunks)', linewidth=2, alpha=0.8)

        # Mark chunk boundaries
        for start_frame, _ in second_chunks_predicted:
            t = start_frame / fps
            ax.axvline(x=t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            if start_frame < 5 * fps:
                ax.text(t + 0.05, ax.get_ylim()[1] * 0.98, f'{int(t)}s',
                       fontsize=7, alpha=0.7, verticalalignment='top')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Action value", fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = cfg.output_dir / "action_comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {plot_path}")

    # Additional diagnostic plot: Show individual actions within a few chunks
    if len(second_chunks_predicted) >= 3:
        fig, axes = plt.subplots(3, action_dim, figsize=(3 * action_dim, 9), squeeze=False)
        fig.suptitle("Individual Actions Within Predicted Chunks (First 3 Chunks)", fontsize=12)

        for chunk_idx in range(min(3, len(second_chunks_predicted))):
            start_frame, pred_chunk = second_chunks_predicted[chunk_idx]
            chunk_time = np.arange(len(pred_chunk)) / fps  # Time within chunk (0 to 1 second)

            for dim in range(action_dim):
                ax = axes[chunk_idx, dim]
                name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"

                # Plot predicted actions within chunk
                ax.plot(chunk_time, pred_chunk[:, dim], 'r-', linewidth=2, label='Predicted')

                # Plot corresponding ground truth chunk
                gt_start = start_frame
                gt_end = min(start_frame + len(pred_chunk), len(gt_actions_episode))
                gt_chunk = gt_actions_episode[gt_start:gt_end, dim]
                gt_time = np.arange(len(gt_chunk)) / fps
                ax.plot(gt_time, gt_chunk, 'b--', linewidth=2, alpha=0.7, label='Ground Truth')

                if chunk_idx == 0:
                    ax.set_title(name, fontsize=10)
                if dim == 0:
                    ax.set_ylabel(f"Chunk {chunk_idx}\n({start_frame//fps}s)", fontsize=9)
                if chunk_idx == 2:
                    ax.set_xlabel("Time in chunk (s)", fontsize=9)
                ax.grid(True, alpha=0.3)
                if chunk_idx == 0 and dim == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = cfg.output_dir / "chunk_detail_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {plot_path}")

    # Error over time plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.plot(time_axis, all_errors, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(y=all_errors.mean(), color='r', linestyle='--',
                label=f'Mean: {all_errors.mean():.4f}')
    ax1.fill_between(time_axis, all_errors, alpha=0.3)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Mean L1 Error (per chunk)")
    ax1.set_title("Action Prediction Error Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
    for dim in range(action_dim):
        name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
        ax2.plot(time_axis, per_dim_errors[dim], color=colors[dim],
                linewidth=1, alpha=0.7, label=name)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("L1 Error")
    ax2.set_title("Per-Dimension Error Over Time")
    ax2.legend(loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = cfg.output_dir / "action_error_over_time.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {plot_path}")

    # Save results JSON
    results = {
        "inference_latency_ms": {
            "mean": float(inference_times.mean() * 1000),
            "std": float(inference_times.std() * 1000),
            "min": float(inference_times.min() * 1000),
            "max": float(inference_times.max() * 1000),
            "median": float(np.median(inference_times) * 1000),
            "p95": float(np.percentile(inference_times, 95) * 1000),
            "p99": float(np.percentile(inference_times, 99) * 1000),
            "effective_hz": float(effective_hz),
        },
        "action_error": {
            "mean": float(all_errors.mean()),
            "std": float(all_errors.std()),
            "max": float(all_errors.max()),
            "min": float(all_errors.min()),
        },
        "per_dimension_error": {
            dim_names[d] if d < len(dim_names) else f"dim_{d}": {
                "mean": float(np.array(per_dim_errors[d]).mean()),
                "max": float(np.array(per_dim_errors[d]).max()),
            }
            for d in range(action_dim)
        },
        "config": {
            "checkpoint_path": str(cfg.checkpoint_path),
            "repo_id": cfg.repo_id,
            "episode_idx": cfg.episode_idx,
            "num_frames": num_frames,
            "fps": fps,
            "chunk_size": chunk_size,
        },
    }

    results_path = cfg.output_dir / "policy_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved: {results_path}")

    logging.info("\n✅ Validation complete!")


@draccus.wrap()
def main(cfg: ValidatePolicyConfig):
    run_validation(cfg)


if __name__ == "__main__":
    main()
