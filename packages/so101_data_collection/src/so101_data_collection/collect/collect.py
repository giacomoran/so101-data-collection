#!/usr/bin/env python
"""
Vanilla data collection script for SO-101.

This is a simplified collection script modeled after lerobot-record CLI.
No frame alignment or latency compensation - just straightforward synchronous recording.

For frame-aligned collection with camera latency compensation, use collect_aligned.py.

Supports collection setups:
- leader_teleop: Standard leader-follower teleoperation
- hand_guided: Single arm provides both observation and action

Usage:
    # Basic collection
    python -m so101_data_collection.collect.collect \
        --task cube \
        --setup hand_guided \
        --num-episodes 50

    # With custom repo ID
    python -m so101_data_collection.collect.collect \
        --task cube \
        --setup hand_guided \
        --repo-id giacomoran/my_custom_dataset \
        --num-episodes 50

    # Resume existing dataset
    python -m so101_data_collection.collect.collect \
        --task cube \
        --setup hand_guided \
        --resume
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from so101_data_collection.shared.setup import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    LEADER_ID,
    LEADER_PORT,
    ROBOT_ID,
    ROBOT_PORT,
    TOP_CAMERA_INDEX,
    WRIST_CAMERA_INDEX,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_NUM_EPISODES = 50
DEFAULT_FPS = 30
DEFAULT_EPISODE_TIME_S = 20.0
DEFAULT_RESET_TIME_S = 10.0
DEFAULT_DATASET_ROOT = Path("data")


class Task(str, Enum):
    CUBE = "cube"
    GBA = "gba"
    BALL = "ball"


class Setup(str, Enum):
    LEADER_TELEOP = "leader_teleop"
    HAND_GUIDED = "hand_guided"


TASK_DESCRIPTIONS = {
    Task.CUBE: "Pick up the cube and place it in the target location",
    Task.GBA: "Press the up arrow on the GBA",
    Task.BALL: "Throw the ping-pong ball into the basket",
}


@dataclass
class CollectConfig:
    """Configuration for vanilla data collection."""

    task: Task
    setup: Setup
    num_episodes: int = DEFAULT_NUM_EPISODES
    dataset_fps: int = DEFAULT_FPS
    episode_time_s: float = DEFAULT_EPISODE_TIME_S
    reset_time_s: float = DEFAULT_RESET_TIME_S
    resume: bool = False

    # Robot config
    robot_port: str = ROBOT_PORT
    robot_id: str = ROBOT_ID
    leader_port: str = LEADER_PORT
    leader_id: str = LEADER_ID

    # Camera config
    wrist_camera_index: int = WRIST_CAMERA_INDEX
    top_camera_index: int | None = TOP_CAMERA_INDEX
    camera_width: int = CAMERA_WIDTH
    camera_height: int = CAMERA_HEIGHT

    # Dataset config
    # If None, defaults to giacomoran/so101_data_collection_{task}_{setup}
    repo_id: str | None = None
    dataset_root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT)

    play_sounds: bool = True


# ============================================================================
# Hardware Setup
# ============================================================================


def create_cameras(config: CollectConfig) -> dict[str, OpenCVCamera]:
    """Create camera instances."""
    cameras = {}

    # Wrist camera (always present)
    wrist_config = OpenCVCameraConfig(
        index_or_path=config.wrist_camera_index,
        fps=None,  # Let camera run at native FPS
        width=config.camera_width,
        height=config.camera_height,
    )
    cameras["wrist"] = OpenCVCamera(wrist_config)

    # Top camera (optional)
    if config.top_camera_index is not None:
        top_config = OpenCVCameraConfig(
            index_or_path=config.top_camera_index,
            fps=None,
            width=config.camera_width,
            height=config.camera_height,
        )
        cameras["top"] = OpenCVCamera(top_config)

    return cameras


def create_hand_guided_arm(config: CollectConfig) -> SO101Leader:
    """Create arm for hand-guided mode (torque disabled for free movement)."""
    leader_config = SO101LeaderConfig(
        port=config.leader_port,
        id=config.leader_id,
        use_degrees=True,
    )
    return SO101Leader(leader_config)


def create_leader_teleop_setup(
    config: CollectConfig,
) -> tuple[SO101Follower, SO101Leader]:
    """Create robot and leader for standard leader teleoperation."""
    follower_config = SO101FollowerConfig(
        port=config.robot_port,
        id=config.robot_id,
        use_degrees=True,
        cameras={},
    )
    follower = SO101Follower(follower_config)

    leader_config = SO101LeaderConfig(
        port=config.leader_port,
        id=config.leader_id,
        use_degrees=True,
    )
    leader = SO101Leader(leader_config)

    return follower, leader


# ============================================================================
# Dataset Features
# ============================================================================


def get_observation_features(cameras: dict[str, OpenCVCamera]) -> dict[str, Any]:
    """Get observation features for dataset creation."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(MOTOR_NAMES),
        }
    }

    for cam_name, cam in cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam.config.height, cam.config.width, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def get_action_features() -> dict[str, Any]:
    """Get action features for dataset creation."""
    return {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(MOTOR_NAMES),
        }
    }


def build_dataset_features(cameras: dict[str, OpenCVCamera]) -> dict[str, Any]:
    """Build complete feature dict for LeRobotDataset."""
    features = {}
    features.update(get_observation_features(cameras))
    features.update(get_action_features())
    return features


# ============================================================================
# Recording Loops
# ============================================================================


def _state_dict_to_array(state_dict: dict[str, float]) -> np.ndarray:
    """Convert state dict {motor.pos: value} to numpy array in MOTOR_NAMES order."""
    values = []
    for motor_name in MOTOR_NAMES:
        key = f"{motor_name}.pos"
        if key in state_dict:
            values.append(state_dict[key])
        else:
            raise KeyError(f"Missing motor key {key}")
    return np.array(values, dtype=np.float32)


def record_loop_hand_guided(
    arm: SO101Leader,
    cameras: dict[str, OpenCVCamera],
    dataset: LeRobotDataset,
    events: dict,
    config: CollectConfig,
    task_description: str,
) -> None:
    """
    Recording loop for hand-guided mode.

    In this mode, the same arm provides both observation.state and action.
    Simple synchronous capture - no latency compensation.
    """
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < config.episode_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Read arm position
        arm_state = arm.get_action()

        # Read cameras
        images = {}
        for cam_name, cam in cameras.items():
            try:
                image = cam.async_read()
                if image is not None:
                    images[cam_name] = image
            except (TimeoutError, Exception) as e:
                logger.warning(f"Camera {cam_name} read failed: {e}")
                continue

        # Build frame dict
        state_array = _state_dict_to_array(arm_state)
        frame: dict[str, Any] = {
            "task": task_description,
            "observation.state": state_array,
            "action": state_array.copy(),  # Same as observation for hand-guided
        }
        for cam_name, image in images.items():
            frame[f"observation.images.{cam_name}"] = image

        dataset.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / config.dataset_fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


def record_loop_leader_teleop(
    robot: SO101Follower,
    leader: SO101Leader,
    cameras: dict[str, OpenCVCamera],
    dataset: LeRobotDataset,
    events: dict,
    config: CollectConfig,
    task_description: str,
) -> None:
    """
    Recording loop for leader teleoperation mode.

    Leader arm provides action, follower arm provides observation.state.
    Simple synchronous capture - no latency compensation.
    """
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < config.episode_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get observation from follower
        obs = robot.get_observation()
        obs_state = {k: v for k, v in obs.items() if k.endswith(".pos")}

        # Get action from leader
        action = leader.get_action()

        # Send action to follower
        robot.send_action(action)

        # Read cameras
        images = {}
        for cam_name, cam in cameras.items():
            try:
                image = cam.async_read()
                if image is not None:
                    images[cam_name] = image
            except (TimeoutError, Exception) as e:
                logger.warning(f"Camera {cam_name} read failed: {e}")
                continue

        # Build frame dict
        frame: dict[str, Any] = {
            "task": task_description,
            "observation.state": _state_dict_to_array(obs_state),
            "action": _state_dict_to_array(action),
        }
        for cam_name, image in images.items():
            frame[f"observation.images.{cam_name}"] = image

        dataset.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / config.dataset_fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


def reset_loop(events: dict, config: CollectConfig) -> None:
    """Wait for environment reset between episodes."""
    timestamp = 0.0
    start_t = time.perf_counter()

    while timestamp < config.reset_time_s:
        if events["exit_early"]:
            events["exit_early"] = False
            break
        precise_sleep(0.1)
        timestamp = time.perf_counter() - start_t


# ============================================================================
# Dataset Management
# ============================================================================


def get_repo_id(config: CollectConfig) -> str:
    """
    Get the repository ID for the dataset.

    If repo_id is specified, use it as-is.
    Otherwise, construct default: giacomoran/so101_data_collection_{task}_{setup}
    """
    if config.repo_id is not None:
        return config.repo_id
    return f"giacomoran/so101_data_collection_{config.task.value}_{config.setup.value}"


def check_and_remove_existing_dataset(dataset_path: Path) -> None:
    """Check if dataset exists and prompt for deletion."""
    if dataset_path.exists():
        logger.warning(f"Dataset directory already exists: {dataset_path}")
        response = input(f"Delete existing dataset directory '{dataset_path}'? [y/N]: ")
        if response.strip().lower() == "y":
            logger.info(f"Deleting {dataset_path}...")
            shutil.rmtree(dataset_path)
            logger.info("Directory deleted successfully.")
        else:
            logger.error("Dataset directory exists. Exiting.")
            sys.exit(1)


# ============================================================================
# Main Collection
# ============================================================================


def collect(config: CollectConfig) -> None:
    """Main vanilla data collection function."""
    init_logging()

    repo_id = get_repo_id(config)
    task_description = TASK_DESCRIPTIONS[config.task]

    logger.info(
        f"Starting collection: task={config.task.value}, setup={config.setup.value}"
    )
    logger.info(f"Dataset: {repo_id}")
    logger.info(f"FPS: {config.dataset_fps}, Episodes: {config.num_episodes}")

    # Create cameras
    cameras = create_cameras(config)

    # Setup dataset path
    dataset_path = config.dataset_root / repo_id

    if config.resume:
        if not dataset_path.exists():
            logger.error(f"Cannot resume: dataset not found: {dataset_path}")
            sys.exit(1)
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        logger.info(f"Resuming dataset with {dataset.num_episodes} episodes")
    else:
        check_and_remove_existing_dataset(dataset_path)
        features = build_dataset_features(cameras)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=config.dataset_fps,
            root=dataset_path,
            robot_type="so101",
            features=features,
            use_videos=True,
        )

    # Setup hardware based on collection mode
    arm = None
    robot = None
    leader = None
    listener = None

    try:
        if config.setup == Setup.HAND_GUIDED:
            arm = create_hand_guided_arm(config)
            arm.connect()
        elif config.setup == Setup.LEADER_TELEOP:
            robot, leader = create_leader_teleop_setup(config)
            robot.connect()
            leader.connect()

        # Connect cameras
        for cam in cameras.values():
            cam.connect()

        # Setup keyboard listener
        listener, events = init_keyboard_listener()

        recorded_episodes = 0

        while recorded_episodes < config.num_episodes and not events["stop_recording"]:
            episode_num = recorded_episodes + 1
            log_say(
                f"Recording episode {episode_num} of {config.num_episodes}",
                config.play_sounds,
            )

            # Record episode
            if config.setup == Setup.HAND_GUIDED:
                record_loop_hand_guided(
                    arm=arm,
                    cameras=cameras,
                    dataset=dataset,
                    events=events,
                    config=config,
                    task_description=task_description,
                )
            elif config.setup == Setup.LEADER_TELEOP:
                record_loop_leader_teleop(
                    robot=robot,
                    leader=leader,
                    cameras=cameras,
                    dataset=dataset,
                    events=events,
                    config=config,
                    task_description=task_description,
                )

            # Handle re-record
            if events["rerecord_episode"]:
                log_say("Re-recording episode", config.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
            logger.info(f"Episode {episode_num} saved")

            # Reset time (skip if done)
            if recorded_episodes < config.num_episodes and not events["stop_recording"]:
                log_say("Reset the environment", config.play_sounds)
                reset_loop(events, config)

        # Finalize dataset
        dataset.finalize()

        log_say("Recording complete", config.play_sounds, blocking=True)
        logger.info(f"Recorded {recorded_episodes} episodes")
        logger.info(f"Dataset saved to: {dataset_path}")

    finally:
        # Cleanup
        if arm is not None and arm.is_connected:
            arm.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()
        if leader is not None and leader.is_connected:
            leader.disconnect()
        for cam in cameras.values():
            if cam.is_connected:
                cam.disconnect()
        if not is_headless() and listener:
            listener.stop()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> CollectConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SO-101 Vanilla Data Collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[t.value for t in Task],
        help="Task to collect data for",
    )
    parser.add_argument(
        "--setup",
        type=str,
        required=True,
        choices=[s.value for s in Setup],
        help="Data collection setup",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--dataset-fps",
        type=int,
        default=DEFAULT_FPS,
        help="Output dataset FPS",
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=DEFAULT_EPISODE_TIME_S,
        help="Maximum episode duration (s)",
    )
    parser.add_argument(
        "--reset-time",
        type=float,
        default=DEFAULT_RESET_TIME_S,
        help="Environment reset time (s)",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default=ROBOT_PORT,
        help="Robot USB port",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default=ROBOT_ID,
        help="Robot ID for calibration",
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        default=LEADER_PORT,
        help="Leader arm USB port",
    )
    parser.add_argument(
        "--leader-id",
        type=str,
        default=LEADER_ID,
        help="Leader ID for calibration",
    )
    parser.add_argument(
        "--wrist-camera",
        type=int,
        default=WRIST_CAMERA_INDEX,
        help="Wrist camera index",
    )
    parser.add_argument(
        "--top-camera",
        type=int,
        default=TOP_CAMERA_INDEX,
        help="Top camera index (omit for wrist-only)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=CAMERA_WIDTH,
        help="Camera capture width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=CAMERA_HEIGHT,
        help="Camera capture height",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (default: giacomoran/so101_data_collection_{task}_{setup})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume recording on existing dataset",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable sound feedback",
    )

    args = parser.parse_args()

    return CollectConfig(
        task=Task(args.task),
        setup=Setup(args.setup),
        num_episodes=args.num_episodes,
        dataset_fps=args.dataset_fps,
        episode_time_s=args.episode_time,
        reset_time_s=args.reset_time,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        leader_port=args.leader_port,
        leader_id=args.leader_id,
        wrist_camera_index=args.wrist_camera,
        top_camera_index=args.top_camera,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        repo_id=args.repo_id,
        resume=args.resume,
        play_sounds=not args.no_sound,
    )


def main() -> None:
    config = parse_args()
    collect(config)


if __name__ == "__main__":
    main()
