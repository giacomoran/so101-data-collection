#!/usr/bin/env python

"""Utility functions for computing relative statistics for ACT Relative RTC.

This module computes normalization statistics on relative values (observation deltas
and relative actions) rather than absolute values. This is necessary because:
- The model computes relative transformations: delta_obs = obs[t] - obs[t-N]
  (where N = obs_state_delta_frames, default 1)
- Actions are relative to the current observation: rel_action = action - obs[t]
- normalize(a) - normalize(b) ≠ normalize(a - b)

Therefore, we need to compute statistics on the relative values themselves.
"""

import logging
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use the same constants as the model
OBS_STATE = "observation.state"
ACTION = "action"


def compute_relative_stats(
    dataset: Any,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute statistics on relative values (delta_obs and relative_actions).

    This function iterates through the dataset and computes the same relative
    transformations as the model's forward() method:
    - delta_obs = obs[t] - obs[t-N]  (where N = obs_state_delta_frames)
    - relative_actions = action - obs[t]

    The statistics are computed over all samples, with relative_actions flattened
    across the temporal (chunk) dimension to match what the model sees.

    Args:
        dataset: LeRobotDataset with delta_timestamps configured to return
                 observation.state with shape [2, state_dim] where:
                 - index 0 = obs[t - obs_state_delta_frames]
                 - index 1 = obs[t]
                 and action with shape [chunk_size, action_dim].
        batch_size: Batch size for iterating through the dataset.
        num_workers: Number of workers for the DataLoader. Defaults to 4.
                    This is controlled by the policy's `num_workers` config parameter,
                    which can be set via `--policy.num_workers` CLI argument.

    Returns:
        Dictionary with statistics for delta_obs and relative_action:
        {
            "delta_obs": {"mean": np.array, "std": np.array},
            "relative_action": {"mean": np.array, "std": np.array}
        }
    """
    logging.info("Computing relative statistics for normalization...")

    delta_obs_list = []
    relative_action_list = []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    for batch in tqdm(dataloader, desc="Computing relative stats"):
        # Extract obs[t-N] and obs[t] from stacked observations
        # obs.state shape: [batch, 2, state_dim]
        obs_state_stacked = batch[OBS_STATE]
        obs_state_t_minus_n = obs_state_stacked[:, 0, :]  # [batch, state_dim]
        obs_state_t = obs_state_stacked[:, 1, :]  # [batch, state_dim]

        # Compute delta observation (same as model.forward())
        delta_obs = obs_state_t - obs_state_t_minus_n  # [batch, state_dim]

        # Compute relative actions (same as model.forward())
        # action shape: [batch, chunk_size, action_dim]
        absolute_actions = batch[ACTION]
        relative_actions = absolute_actions - obs_state_t.unsqueeze(1)  # [batch, chunk_size, action_dim]

        # Get action padding mask if available (from LeRobot's delta_timestamps handling)
        # action_is_pad shape: [batch, chunk_size] - True where action is padded
        action_is_pad = batch.get(f"{ACTION}_is_pad", None)

        # Flatten relative_actions across batch and temporal dimensions
        # This treats all timesteps equally for statistics computation
        batch_size_actual, chunk_size, action_dim = relative_actions.shape
        relative_actions_flat = relative_actions.reshape(-1, action_dim)  # [batch * chunk_size, action_dim]

        # Filter out padded actions if padding mask is available
        if action_is_pad is not None:
            action_is_pad_flat = action_is_pad.reshape(-1)  # [batch * chunk_size]
            # Only keep non-padded actions
            relative_actions_flat = relative_actions_flat[~action_is_pad_flat]

        delta_obs_list.append(delta_obs.numpy())
        if len(relative_actions_flat) > 0:
            relative_action_list.append(relative_actions_flat.numpy())

    # Concatenate all batches
    all_delta_obs = np.concatenate(delta_obs_list, axis=0)  # [N, state_dim]
    all_relative_actions = np.concatenate(relative_action_list, axis=0)  # [N * chunk_size, action_dim]

    # Compute statistics
    delta_obs_stats = {
        "mean": np.mean(all_delta_obs, axis=0).astype(np.float32),
        "std": np.std(all_delta_obs, axis=0).astype(np.float32),
    }
    relative_action_stats = {
        "mean": np.mean(all_relative_actions, axis=0).astype(np.float32),
        "std": np.std(all_relative_actions, axis=0).astype(np.float32),
    }

    logging.info(f"Delta obs stats - mean: {delta_obs_stats['mean']}, std: {delta_obs_stats['std']}")
    logging.info(f"Relative action stats - mean: {relative_action_stats['mean']}, std: {relative_action_stats['std']}")

    return {
        "delta_obs": delta_obs_stats,
        "relative_action": relative_action_stats,
    }
