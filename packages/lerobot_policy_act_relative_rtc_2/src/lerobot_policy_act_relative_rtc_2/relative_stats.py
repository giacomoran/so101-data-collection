#!/usr/bin/env python

"""Utility functions for computing relative statistics for ACT Relative RTC v2.

This module computes normalization statistics on relative actions rather than absolute values.
This is necessary because:
- Actions are relative to the current observation: rel_action = action - obs[t]
- normalize(a) - normalize(b) ≠ normalize(a - b)

Therefore, we need to compute statistics on the relative values themselves.

V2 changes:
- Removed delta_obs computation (no longer used)
- Load single observation instead of two
- Handle extended action sequence for action prefix conditioning
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
    """Compute statistics on relative actions.

    V2 change: Only computes relative action statistics (delta_obs removed).

    This function iterates through the dataset and computes the same relative
    transformations as the model's forward() method:
    - relative_actions = action - obs[t]

    The statistics are computed over all samples, with relative_actions flattened
    across the temporal dimension to match what the model sees.

    Args:
        dataset: LeRobotDataset with delta_timestamps configured to return
                 observation.state with shape [batch, state_dim] and
                 action with shape [batch, rtc_max_delay + chunk_size, action_dim].
        batch_size: Batch size for iterating through the dataset.
        num_workers: Number of workers for the DataLoader. Defaults to 4.
                    This is controlled by the policy's `num_workers` config parameter,
                    which can be set via `--policy.num_workers` CLI argument.

    Returns:
        Dictionary with statistics for relative_action only:
        {
            "relative_action": {"mean": np.array, "std": np.array}
        }
    """
    logging.info("Computing relative statistics for normalization...")

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
        # V2 change: Load single observation
        # obs.state shape: [batch, state_dim]
        obs_state_t = batch[OBS_STATE]

        # Compute relative actions (same as model.forward())
        # V2 change: action shape is now [batch, rtc_max_delay + chunk_size, action_dim]
        absolute_actions = batch[ACTION]
        relative_actions = absolute_actions - obs_state_t.unsqueeze(1)  # [batch, seq_len, action_dim]

        # Get action padding mask if available (from LeRobot's delta_timestamps handling)
        # V2 change: action_is_pad shape is now [batch, rtc_max_delay + chunk_size]
        action_is_pad = batch.get(f"{ACTION}_is_pad", None)

        # Flatten relative_actions across batch and temporal dimensions
        # This treats all timesteps equally for statistics computation
        batch_size_actual, seq_len, action_dim = relative_actions.shape
        relative_actions_flat = relative_actions.reshape(-1, action_dim)  # [batch * seq_len, action_dim]

        # Filter out padded actions if padding mask is available
        if action_is_pad is not None:
            action_is_pad_flat = action_is_pad.reshape(-1)  # [batch * seq_len]
            # Only keep non-padded actions
            relative_actions_flat = relative_actions_flat[~action_is_pad_flat]

        if len(relative_actions_flat) > 0:
            relative_action_list.append(relative_actions_flat.numpy())

    # Concatenate all batches
    all_relative_actions = np.concatenate(relative_action_list, axis=0)  # [N * seq_len, action_dim]

    # Compute statistics
    relative_action_stats = {
        "mean": np.mean(all_relative_actions, axis=0).astype(np.float32),
        "std": np.std(all_relative_actions, axis=0).astype(np.float32),
    }

    logging.info(f"Relative action stats - mean: {relative_action_stats['mean']}, std: {relative_action_stats['std']}")

    return {
        "relative_action": relative_action_stats,
    }
