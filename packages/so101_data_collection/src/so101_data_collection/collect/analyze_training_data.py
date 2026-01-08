#!/usr/bin/env python
"""Analyze training data for ACT Relative RTC policy.

This script visualizes the distributions of:
1. Observation delta proprioception (obs[t] - obs[t-1]) for each joint
2. Relative actions (action[t] - obs[t]) for each joint
3. Action chunks of relative joint positions at each timestamp

The computations match exactly what's done in modeling_act_relative_rtc.py:
- delta_obs = obs[t] - obs[t-1]
- relative_action = action - obs[t]

Usage:
    python -m so101_data_collection.collect.analyze_training_data \
        --dataset-repo-id=giacomoran/so101_data_collection_cube_hand_guided \
        --dataset-episode-idx=[0,1,2,3,4]
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import init_logging

# Joint names for SO-101 (6 DOF)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class AnalyzeTrainingDataConfig:
    """Configuration for training data analysis."""

    dataset_repo_id: str = "giacomoran/so101_data_collection_cube_hand_guided"
    dataset_episode_idx: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    chunk_size: int = 30  # Default chunk size for ACT


def load_episode_with_deltas(
    dataset: LeRobotDataset,
    episode_idx: int,
    ds_meta: LeRobotDatasetMetadata,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Load a single episode and compute delta_obs and relative_actions.

    This matches the computations in modeling_act_relative_rtc.py:
    - delta_obs = obs[t] - obs[t-1]
    - relative_action = action - obs[t]

    Returns:
        Dictionary containing:
        - obs_state: [num_frames, state_dim] absolute observations
        - delta_obs: [num_frames, state_dim] observation deltas
        - action_chunks: [num_frames, chunk_size, action_dim] absolute action chunks
        - relative_action_chunks: [num_frames, chunk_size, action_dim] relative action chunks
        - relative_actions_single: [num_frames, action_dim] relative actions (first timestep only)
    """
    fps = ds_meta.fps
    state_dim = len(JOINT_NAMES)
    action_dim = state_dim  # SO-101 has 6-DOF for both state and action

    # Build delta_timestamps matching ACT Relative RTC configuration
    # We need obs[t-1] and obs[t] for delta computation
    delta_timestamps = {
        "observation.state": [-1 / fps, 0],
        "observation.images.wrist": [-1 / fps, 0],
        "observation.images.top": [-1 / fps, 0],
        "action": [i / fps for i in range(chunk_size)],
    }

    # Create dataset with appropriate delta_timestamps
    dataset_with_deltas = LeRobotDataset(
        ds_meta.repo_id,
        root=ds_meta.root,
        episodes=None,  # Load all episodes, we'll index manually
        delta_timestamps=delta_timestamps,
    )

    # Get episode bounds
    from_idx = ds_meta.episodes["dataset_from_index"][episode_idx]
    to_idx = ds_meta.episodes["dataset_to_index"][episode_idx]
    from_idx = int(from_idx.item() if hasattr(from_idx, "item") else from_idx)
    to_idx = int(to_idx.item() if hasattr(to_idx, "item") else to_idx)
    num_frames = to_idx - from_idx

    logging.info(
        f"Episode {episode_idx}: {num_frames} frames at {fps} fps, "
        f"dataset indices: [{from_idx}, {to_idx})"
    )

    # Storage
    obs_state_list = []
    delta_obs_list = []
    action_chunks_list = []
    relative_action_chunks_list = []
    relative_actions_single_list = []

    # Process each frame
    for local_idx in range(num_frames):
        global_idx = from_idx + local_idx
        sample = dataset_with_deltas[global_idx]

        # Extract obs.state stacked: [2, state_dim] containing [obs[t-1], obs[t]]
        obs_state_stacked = sample["observation.state"]
        if isinstance(obs_state_stacked, torch.Tensor):
            obs_state_stacked = obs_state_stacked.cpu().numpy()

        # obs[t-1] and obs[t]
        obs_t_minus_1 = obs_state_stacked[0]  # [state_dim]
        obs_t = obs_state_stacked[1]  # [state_dim]

        # Compute delta_obs = obs[t] - obs[t-1] (matches modeling_act_relative_rtc.py line 391)
        delta_obs = obs_t - obs_t_minus_1  # [state_dim]

        # Extract action chunk: [chunk_size, action_dim]
        action_chunk = sample["action"]
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()

        # Compute relative_action = action - obs[t] (matches modeling_act_relative_rtc.py line 397)
        # obs_t shape: [state_dim] -> [1, state_dim] for broadcasting
        # action_chunk shape: [chunk_size, action_dim]
        relative_action_chunk = action_chunk - obs_t[np.newaxis, :]  # [chunk_size, action_dim]

        # Store
        obs_state_list.append(obs_t)
        delta_obs_list.append(delta_obs)
        action_chunks_list.append(action_chunk)
        relative_action_chunks_list.append(relative_action_chunk)
        relative_actions_single_list.append(relative_action_chunk[0])  # First timestep only

    return {
        "obs_state": np.array(obs_state_list),  # [num_frames, state_dim]
        "delta_obs": np.array(delta_obs_list),  # [num_frames, state_dim]
        "action_chunks": np.array(action_chunks_list),  # [num_frames, chunk_size, action_dim]
        "relative_action_chunks": np.array(relative_action_chunks_list),  # [num_frames, chunk_size, action_dim]
        "relative_actions_single": np.array(relative_actions_single_list),  # [num_frames, action_dim]
    }


def plot_delta_obs_distribution(
    all_delta_obs: np.ndarray,
    episode_indices: list[int],
) -> plt.Figure:
    """Plot distribution of observation deltas for each joint.

    Args:
        all_delta_obs: [total_frames, state_dim] concatenated delta observations from all episodes
        episode_indices: List of episode indices being analyzed

    Returns:
        Figure object
    """
    num_joints = len(JOINT_NAMES)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle(
        f"Distribution of Observation Deltas (obs[t] - obs[t-1])\n"
        f"Episodes: {episode_indices}, Total Frames: {len(all_delta_obs)}",
        fontsize=14,
    )

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        joint_name = JOINT_NAMES[joint_idx]
        deltas = all_delta_obs[:, joint_idx]

        # Histogram with KDE
        ax.hist(
            deltas,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Statistics
        mean = deltas.mean()
        std = deltas.std()
        median = np.median(deltas)
        p5, p95 = np.percentile(deltas, [5, 95])
        min_val, max_val = deltas.min(), deltas.max()

        ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.4f}")
        ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median: {median:.4f}")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Delta (rad)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(joint_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add stats text box
        stats_text = (
            f"μ: {mean:.4f}\n"
            f"σ: {std:.4f}\n"
            f"5-95%: [{p5:.4f}, {p95:.4f}]\n"
            f"Range: [{min_val:.4f}, {max_val:.4f}]"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig


def plot_relative_action_distribution(
    all_relative_action_chunks: np.ndarray,
    all_relative_actions_single: np.ndarray,
    episode_indices: list[int],
    chunk_size: int,
) -> plt.Figure:
    """Plot distribution of relative actions for each joint.

    Shows both:
    1. All actions across all timesteps in action chunks
    2. Actions at individual timesteps (first timestep only)

    Args:
        all_relative_action_chunks: [total_frames, chunk_size, action_dim] relative action chunks
        all_relative_actions_single: [total_frames, action_dim] relative actions (first timestep only)
        episode_indices: List of episode indices being analyzed
        chunk_size: Size of action chunks

    Returns:
        Figure object
    """
    num_joints = len(JOINT_NAMES)

    # Flatten across chunk dimension for "all actions" view
    # [total_frames, chunk_size, action_dim] -> [total_frames * chunk_size, action_dim]
    all_actions_flattened = all_relative_action_chunks.reshape(-1, num_joints)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle(
        f"Distribution of Relative Actions (action[t] - obs[t])\n"
        f"Episodes: {episode_indices}, Total Frames: {len(all_relative_action_chunks)}\n"
        f"Blue: All {chunk_size} timesteps in chunk | Red: First timestep only",
        fontsize=14,
    )

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        joint_name = JOINT_NAMES[joint_idx]

        # All actions across chunk
        all_actions = all_actions_flattened[:, joint_idx]
        # Single timestep actions
        single_actions = all_relative_actions_single[:, joint_idx]

        # Plot both distributions
        ax.hist(
            single_actions,
            bins=50,
            density=True,
            alpha=0.5,
            color="red",
            label=f"Single timestep (n={len(single_actions)})",
        )
        ax.hist(
            all_actions,
            bins=50,
            density=True,
            alpha=0.5,
            color="steelblue",
            label=f"All chunk timesteps (n={len(all_actions)})",
        )

        # Statistics for single timestep
        mean_single = single_actions.mean()
        std_single = single_actions.std()

        # Statistics for all timesteps
        mean_all = all_actions.mean()
        std_all = all_actions.std()

        ax.axvline(mean_single, color="red", linestyle="--", linewidth=2)
        ax.axvline(mean_all, color="steelblue", linestyle="--", linewidth=2)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Relative Action (rad)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(joint_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add stats text box
        stats_text = (
            f"Single timestep:\n"
            f"  μ: {mean_single:.4f}, σ: {std_single:.4f}\n\n"
            f"All timesteps:\n"
            f"  μ: {mean_all:.4f}, σ: {std_all:.4f}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig


def plot_relative_action_chunks_over_time(
    episode_data: dict[str, np.ndarray],
    episode_idx: int,
    chunk_size: int,
) -> plt.Figure:
    """Plot relative action chunks at each timestamp in an episode.

    For each joint, shows the full action chunk (30 timesteps) at each timestamp.

    Args:
        episode_data: Dictionary containing episode data from load_episode_with_deltas
        episode_idx: Episode index being plotted
        chunk_size: Size of action chunks

    Returns:
        Figure object
    """
    relative_action_chunks = episode_data["relative_action_chunks"]  # [num_frames, chunk_size, action_dim]
    num_frames, _, num_joints = relative_action_chunks.shape

    # Color map for chunks
    colors = plt.cm.viridis(np.linspace(0, 1, num_frames))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle(
        f"Relative Action Chunks Over Time (Episode {episode_idx})\n"
        f"Each line is the full {chunk_size}-step action chunk at a timestamp",
        fontsize=14,
    )

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        joint_name = JOINT_NAMES[joint_idx]

        # Plot each chunk
        for frame_idx in range(num_frames):
            chunk = relative_action_chunks[frame_idx, :, joint_idx]
            chunk_time = np.arange(chunk_size)
            ax.plot(
                chunk_time,
                chunk,
                color=colors[frame_idx],
                alpha=0.6,
                linewidth=0.8,
            )

        ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Timestep in Chunk", fontsize=10)
        ax.set_ylabel("Relative Action (rad)", fontsize=10)
        ax.set_title(joint_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add stats
        all_values = relative_action_chunks[:, :, joint_idx].flatten()
        mean = all_values.mean()
        std = all_values.std()
        ax.text(
            0.98,
            0.02,
            f"μ: {mean:.4f}, σ: {std:.4f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig


def plot_relative_action_chunks_first_k(
    all_episode_data: list[dict[str, np.ndarray]],
    episode_indices: list[int],
    chunk_size: int,
    max_chunks_per_episode: int = 10,
) -> plt.Figure:
    """Plot first K action chunks from each episode to see variation across episodes.

    This helps visualize how chunks evolve at the beginning of episodes.

    Args:
        all_episode_data: List of episode data dictionaries
        episode_indices: List of episode indices
        chunk_size: Size of action chunks
        max_chunks_per_episode: Maximum number of chunks to plot per episode

    Returns:
        Figure object
    """
    num_joints = len(JOINT_NAMES)
    num_episodes = len(episode_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle(
        f"First {max_chunks_per_episode} Action Chunks per Episode\n"
        f"Episodes: {episode_indices}",
        fontsize=14,
    )

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        joint_name = JOINT_NAMES[joint_idx]

        for ep_idx, (ep_data, color) in enumerate(zip(all_episode_data, colors)):
            relative_action_chunks = ep_data["relative_action_chunks"]
            num_chunks_to_plot = min(max_chunks_per_episode, len(relative_action_chunks))

            for chunk_idx in range(num_chunks_to_plot):
                chunk = relative_action_chunks[chunk_idx, :, joint_idx]
                chunk_time = np.arange(chunk_size)

                # Plot with fading alpha for later chunks
                alpha = 0.8 - (chunk_idx / num_chunks_to_plot) * 0.5
                ax.plot(
                    chunk_time,
                    chunk,
                    color=color,
                    alpha=alpha,
                    linewidth=0.8,
                    label=f"Ep {episode_indices[ep_idx]}" if chunk_idx == 0 else None,
                )

        ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Timestep in Chunk", fontsize=10)
        ax.set_ylabel("Relative Action (rad)", fontsize=10)
        ax.set_title(joint_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    return fig


def run_analysis(cfg: AnalyzeTrainingDataConfig):
    """Run training data analysis."""
    init_logging()

    logging.info("=" * 80)
    logging.info("ACT Relative RTC Training Data Analysis")
    logging.info("=" * 80)
    logging.info(f"Dataset: {cfg.dataset_repo_id}")
    logging.info(f"Episodes: {cfg.dataset_episode_idx}")
    logging.info(f"Chunk size: {cfg.chunk_size}")
    logging.info("")

    # Load dataset metadata
    ds_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)
    logging.info(f"Dataset FPS: {ds_meta.fps}")
    logging.info(f"Total episodes: {ds_meta.total_episodes}")
    logging.info(f"Total frames: {ds_meta.total_frames}")
    logging.info("")

    # Load and analyze each episode
    all_episode_data = []
    all_delta_obs = []
    all_relative_action_chunks = []
    all_relative_actions_single = []

    for episode_idx in cfg.dataset_episode_idx:
        episode_data = load_episode_with_deltas(
            None,  # dataset parameter not needed since we recreate in the function
            episode_idx,
            ds_meta,
            cfg.chunk_size,
        )
        all_episode_data.append(episode_data)

        # Concatenate data for distribution plots
        all_delta_obs.append(episode_data["delta_obs"])
        all_relative_action_chunks.append(episode_data["relative_action_chunks"])
        all_relative_actions_single.append(episode_data["relative_actions_single"])

        logging.info(
            f"  Episode {episode_idx}: {len(episode_data['delta_obs'])} frames"
        )
        logging.info(
            f"    Delta obs range: [{episode_data['delta_obs'].min():.4f}, "
            f"{episode_data['delta_obs'].max():.4f}]"
        )
        logging.info(
            f"    Relative action range: [{episode_data['relative_action_chunks'].min():.4f}, "
            f"{episode_data['relative_action_chunks'].max():.4f}]"
        )
        logging.info("")

    # Concatenate across episodes
    all_delta_obs = np.concatenate(all_delta_obs, axis=0)  # [total_frames, state_dim]
    all_relative_action_chunks = np.concatenate(all_relative_action_chunks, axis=0)  # [total_frames, chunk_size, action_dim]
    all_relative_actions_single = np.concatenate(all_relative_actions_single, axis=0)  # [total_frames, action_dim]

    logging.info(f"Total frames across all episodes: {len(all_delta_obs)}")
    logging.info("")

    # Generate plots
    logging.info("📊 Generating plots...")

    # Plot 1: Delta observation distribution
    logging.info("  Plotting delta observation distribution...")
    fig1 = plot_delta_obs_distribution(all_delta_obs, cfg.dataset_episode_idx)

    # Plot 2: Relative action distribution
    logging.info("  Plotting relative action distribution...")
    fig2 = plot_relative_action_distribution(
        all_relative_action_chunks,
        all_relative_actions_single,
        cfg.dataset_episode_idx,
        cfg.chunk_size,
    )

    # Plot 3: Relative action chunks for first episode (overview)
    logging.info("  Plotting relative action chunks for first episode...")
    fig3 = plot_relative_action_chunks_over_time(
        all_episode_data[0],
        cfg.dataset_episode_idx[0],
        cfg.chunk_size,
    )

    # Plot 4: First K chunks per episode (cross-episode comparison)
    logging.info("  Plotting first K chunks per episode...")
    fig4 = plot_relative_action_chunks_first_k(
        all_episode_data,
        cfg.dataset_episode_idx,
        cfg.chunk_size,
        max_chunks_per_episode=10,
    )

    # Show all plots
    logging.info("")
    logging.info("✅ Analysis complete! Displaying plots...")
    plt.show()


def parse_args() -> AnalyzeTrainingDataConfig:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze training data for ACT Relative RTC policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="giacomoran/so101_data_collection_cube_hand_guided",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--dataset-episode-idx",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Episode indices to analyze",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Action chunk size (should match model config)",
    )

    args = parser.parse_args()
    return AnalyzeTrainingDataConfig(
        dataset_repo_id=args.dataset_repo_id,
        dataset_episode_idx=args.dataset_episode_idx,
        chunk_size=args.chunk_size,
    )


def main():
    cfg = parse_args()
    run_analysis(cfg)


if __name__ == "__main__":
    main()

