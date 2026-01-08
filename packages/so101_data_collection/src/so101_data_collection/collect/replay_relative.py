#!/usr/bin/env python

"""
Replays the actions of an episode from a dataset on a robot with relative joint processing.

This script mirrors the lerobot-replay functionality but converts absolute actions to relative,
normalizes them, then unnormalizes and converts back to absolute before sending to the robot.
This helps test whether the normalization/denormalization pipeline introduces issues.

Usage examples:

1. Fixed interval mode (default N=1):
```bash
python -m so101_data_collection.collect.replay_relative \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A460829821 \
  --robot.id=arm_follower_0 \
  --dataset.repo_id=giacomoran/so101_data_collection_cube_hand_guided \
  --dataset.episode=0 \
  --relative_fixed_interval=1 \
  --policy.pretrained_path=/path/to/policy
```

2. Sliding interval mode (re-anchor per chunk):
```bash
python -m so101_data_collection.collect.replay_relative \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A460829821 \
  --robot.id=arm_follower_0 \
  --dataset.repo_id=giacomoran/so101_data_collection_cube_hand_guided \
  --dataset.episode=0 \
  --relative_sliding_interval=100 \
  --policy.pretrained_path=/path/to/policy
```

The two modes:
- `--relative_fixed_interval=N`: For each timestep t, compute relative action w.r.t. observation at t-N.
  If t < N, use the earliest valid observation (t=0).
- `--relative_sliding_interval=N`: Fix an anchor observation, roll out the next N actions relative
  to that anchor, then advance the anchor and repeat. This mirrors inference-time chunk behavior.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Literal

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class PolicyConfig:
    # Path to pretrained policy checkpoint to extract relative stats from.
    pretrained_path: str | None = None


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    policy: PolicyConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Relative mode: "fixed" or "sliding"
    relative_mode: Literal["fixed", "sliding"] = "fixed"
    # For fixed mode: compute relative actions w.r.t. observation N frames back
    relative_fixed_interval: int | None = None
    # For sliding mode: re-anchor every N actions (mirrors chunk_size)
    relative_sliding_interval: int | None = None


class RelativeNormalizer:
    """Normalizer for relative values (mirrors the one in ACTRelativeRTCPolicy).

    This is a simplified version of the Normalizer class that doesn't need to be a PyTorch module.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        self.eps = eps
        self.mean = np.zeros(dim, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32)
        self._is_configured = False

    @property
    def is_configured(self) -> bool:
        """Check if normalization statistics have been set."""
        return self._is_configured

    def configure(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set normalization statistics."""
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self._is_configured = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize: (x - mean) / (std + eps)"""
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Unnormalize: x * std + mean"""
        return x * self.std + self.mean


def load_relative_stats_from_policy(
    pretrained_path: str, device: str
) -> tuple[RelativeNormalizer, RelativeNormalizer]:
    """Load relative stats from a pretrained policy checkpoint.

    Args:
        pretrained_path: Path to pretrained policy checkpoint directory (local or HuggingFace repo ID).
        device: Device to load policy on.

    Returns:
        Tuple of (delta_obs_normalizer, relative_action_normalizer).
    """
    path = Path(pretrained_path)

    # Check if it's a local path or HuggingFace repo ID
    is_local_path = path.exists() or (
        path.parent.exists() and (path / "config.json").exists()
    )

    # Determine policy type from config
    if is_local_path:
        # Local path - check if it's a directory with config.json
        if path.is_dir():
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

        policy_path = str(path)
    else:
        # HuggingFace repo ID - download config to determine policy type
        from huggingface_hub import hf_hub_download

        # Download config.json to determine policy type
        config_file = hf_hub_download(
            repo_id=pretrained_path,
            filename="config.json",
        )

        with open(config_file) as f:
            config_dict = json.load(f)

        policy_type = config_dict.get("type", None)
        if policy_type is None:
            raise ValueError(f"Config from {pretrained_path} missing 'type' field")

        policy_path = pretrained_path

    # Load policy using from_pretrained()
    if policy_type == "act_relative_rtc":
        # For act_relative_rtc, use direct import
        from lerobot_policy_act_relative_rtc import ACTRelativeRTCPolicy

        policy = ACTRelativeRTCPolicy.from_pretrained(
            policy_path,
            device=device,
        )
    else:
        # Use lerobot's factory for standard policies
        from lerobot.policies.factory import get_policy_class

        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(
            policy_path,
            device=device,
        )

    if not policy.has_relative_stats:
        raise ValueError(
            f"Policy at {pretrained_path} does not have relative stats configured. "
            "This script requires a policy that has been trained with relative stats."
        )

    # Extract normalization stats from policy's normalizers
    state_dim = policy.config.robot_state_feature.shape[0]
    action_dim = policy.config.action_feature.shape[0]

    delta_obs_normalizer = RelativeNormalizer(state_dim)
    delta_obs_normalizer.configure(
        mean=policy.delta_obs_normalizer.mean.cpu().numpy(),
        std=policy.delta_obs_normalizer.std.cpu().numpy(),
    )

    relative_action_normalizer = RelativeNormalizer(action_dim)
    relative_action_normalizer.configure(
        mean=policy.relative_action_normalizer.mean.cpu().numpy(),
        std=policy.relative_action_normalizer.std.cpu().numpy(),
    )

    source = "local path" if is_local_path else "HuggingFace"
    logging.info(f"Loaded {policy_type} policy from {source}: {policy_path}")
    logging.info(f"Delta obs mean: {delta_obs_normalizer.mean}")
    logging.info(f"Delta obs std: {delta_obs_normalizer.std}")
    logging.info(f"Relative action mean: {relative_action_normalizer.mean}")
    logging.info(f"Relative action std: {relative_action_normalizer.std}")

    return delta_obs_normalizer, relative_action_normalizer


def process_actions_fixed_interval(
    actions: np.ndarray,
    observations: np.ndarray,
    delta_obs_normalizer: RelativeNormalizer,
    relative_action_normalizer: RelativeNormalizer,
    interval: int,
) -> np.ndarray:
    """Process actions with fixed interval mode.

    For each timestep t:
    - Compute relative action w.r.t. observation at t - interval
    - Normalize it
    - Unnormalize it
    - Convert back to absolute

    Args:
        actions: Array of shape [num_frames, action_dim] with absolute actions.
        observations: Array of shape [num_frames, state_dim] with observation states.
        delta_obs_normalizer: Normalizer for observation deltas (unused in fixed mode).
        relative_action_normalizer: Normalizer for relative actions.
        interval: Number of frames back for computing relative actions.

    Returns:
        Processed absolute actions of shape [num_frames, action_dim].
    """
    num_frames, action_dim = actions.shape
    processed_actions = np.zeros_like(actions)

    for t in range(num_frames):
        # Get anchor observation (t - interval, clamped to 0)
        anchor_idx = max(0, t - interval)
        obs_anchor = observations[anchor_idx]

        # Compute relative action: action[t] - obs[anchor]
        relative_action = actions[t] - obs_anchor

        # Normalize
        relative_action_normalized = relative_action_normalizer.forward(relative_action)

        # Unnormalize
        relative_action_denormalized = relative_action_normalizer.inverse(
            relative_action_normalized
        )

        # Convert back to absolute: relative_action + obs_anchor
        # Must use same anchor we subtracted from
        processed_actions[t] = relative_action_denormalized + obs_anchor

    return processed_actions


def process_actions_sliding_interval(
    actions: np.ndarray,
    observations: np.ndarray,
    delta_obs_normalizer: RelativeNormalizer,
    relative_action_normalizer: RelativeNormalizer,
    interval: int,
) -> np.ndarray:
    """Process actions with sliding interval mode.

    Fix an anchor observation, roll out the next N actions relative to that anchor,
    then advance the anchor and repeat. This mirrors inference-time chunk behavior.

    Args:
        actions: Array of shape [num_frames, action_dim] with absolute actions.
        observations: Array of shape [num_frames, state_dim] with observation states.
        delta_obs_normalizer: Normalizer for observation deltas (unused in sliding mode).
        relative_action_normalizer: Normalizer for relative actions.
        interval: Number of actions to process per anchor (mirrors chunk_size).

    Returns:
        Processed absolute actions of shape [num_frames, action_dim].
    """
    num_frames, action_dim = actions.shape
    processed_actions = np.zeros_like(actions)

    anchor_idx = 0
    while anchor_idx < num_frames:
        # Use observation at anchor_idx as the anchor
        obs_anchor = observations[anchor_idx]

        # Process the next 'interval' actions relative to this anchor
        end_idx = min(anchor_idx + interval, num_frames)

        for t in range(anchor_idx, end_idx):
            # Compute relative action: action[t] - obs[anchor]
            relative_action = actions[t] - obs_anchor

            # Normalize
            relative_action_normalized = relative_action_normalizer.forward(
                relative_action
            )

            # Unnormalize
            relative_action_denormalized = relative_action_normalizer.inverse(
                relative_action_normalized
            )

            # Convert back to absolute: relative_action + obs_anchor
            # Must use same anchor we subtracted from
            processed_actions[t] = relative_action_denormalized + obs_anchor

        # Advance anchor
        anchor_idx = end_idx

    return processed_actions


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Validate configuration
    if cfg.policy.pretrained_path is None:
        raise ValueError(
            "--policy.pretrained_path is required to load relative stats for normalization."
        )

    if cfg.relative_mode == "fixed":
        if cfg.relative_fixed_interval is None:
            cfg.relative_fixed_interval = 1
        interval = cfg.relative_fixed_interval
        logging.info(f"Using fixed interval mode with N={interval}")
    elif cfg.relative_mode == "sliding":
        if cfg.relative_sliding_interval is None:
            cfg.relative_sliding_interval = 100
        interval = cfg.relative_sliding_interval
        logging.info(f"Using sliding interval mode with N={interval}")
    else:
        raise ValueError(f"Unknown relative mode: {cfg.relative_mode}")

    # Auto-select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")

    # Load relative stats from policy
    delta_obs_normalizer, relative_action_normalizer = load_relative_stats_from_policy(
        cfg.policy.pretrained_path, device
    )

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode]
    )

    # Filter dataset to only include frames from the specified episode
    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == cfg.dataset.episode
    )

    # Extract actions and observations
    actions_array = np.array([frame[ACTION] for frame in episode_frames])
    observations_array = np.array([frame[OBS_STATE] for frame in episode_frames])

    logging.info(
        f"Loaded {len(episode_frames)} frames from episode {cfg.dataset.episode}"
    )
    logging.info(f"Actions shape: {actions_array.shape}")
    logging.info(f"Observations shape: {observations_array.shape}")

    # Process actions: convert to relative, normalize, unnormalize, convert back to absolute
    if cfg.relative_mode == "fixed":
        processed_actions = process_actions_fixed_interval(
            actions=actions_array,
            observations=observations_array,
            delta_obs_normalizer=delta_obs_normalizer,
            relative_action_normalizer=relative_action_normalizer,
            interval=interval,
        )
    else:  # sliding
        processed_actions = process_actions_sliding_interval(
            actions=actions_array,
            observations=observations_array,
            delta_obs_normalizer=delta_obs_normalizer,
            relative_action_normalizer=relative_action_normalizer,
            interval=interval,
        )

    # Log statistics about the conversion
    action_diff = np.abs(processed_actions - actions_array)
    logging.info("Action conversion statistics:")
    logging.info(f"  Mean absolute difference: {np.mean(action_diff):.6f}")
    logging.info(f"  Max absolute difference: {np.max(action_diff):.6f}")
    logging.info(f"  Std of absolute difference: {np.std(action_diff):.6f}")

    robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(len(episode_frames)):
        start_episode_t = time.perf_counter()

        action_array = processed_actions[idx]
        action = {}
        for i, name in enumerate(dataset.features[ACTION]["names"]):
            action[name] = action_array[i]

        robot_obs = robot.get_observation()

        processed_action = robot_action_processor((action, robot_obs))

        _ = robot.send_action(processed_action)

        dt_s = time.perf_counter() - start_episode_t
        precise_sleep(1 / dataset.fps - dt_s)

    robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
