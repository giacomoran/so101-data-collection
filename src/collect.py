#!/usr/bin/env python
"""
Data collection script for SO-101 benchmark.

Supports three collection setups:
- phone_teleop: Phone controls end-effector
- leader_teleop: Standard leader-follower teleoperation
- hand_guided: Single arm provides both observation and action

Two collection modes:
1. Regular mode (--num-episodes): Collect N episodes, flush to disk periodically
2. Benchmark mode (--benchmark): Time-capped, all data in RAM, encode after collection

Usage:
    # Regular mode: collect 50 episodes
    python -m src.collect \
        --task cube \
        --setup hand_guided \
        --num-episodes 50

    # Benchmark mode: collect for 15 minutes (900s), no encoding during collection
    python -m src.collect \
        --task cube \
        --setup hand_guided \
        --benchmark 900
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

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402
from lerobot.cameras.opencv import OpenCVCamera  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.robots.so101_follower import SO101Follower  # noqa: E402
from lerobot.robots.so101_follower.config_so101_follower import (  # noqa: E402
    SO101FollowerConfig,
)
from lerobot.teleoperators.so101_leader import SO101Leader  # noqa: E402
from lerobot.teleoperators.so101_leader.config_so101_leader import (  # noqa: E402
    SO101LeaderConfig,
)
from lerobot.utils.control_utils import (  # noqa: E402
    init_keyboard_listener,
    is_headless,
)
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging, log_say  # noqa: E402

from src.benchmark_tracker import BenchmarkTracker, SessionMetrics  # noqa: E402
from src.setup import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    HF_REPO_ID,
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
# Configuration Defaults
# ============================================================================

# Motor order for observation.state and action arrays (must match features definition)
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default collection parameters
DEFAULT_NUM_EPISODES: int = 50
DEFAULT_BENCHMARK_TIME_S: float = 900.0  # 15 minutes per condition (UMI paper approach)
DEFAULT_BUFFER_SIZE: int = 10
DEFAULT_FPS: int = 30
DEFAULT_EPISODE_TIME_S: float = 20.0
DEFAULT_RESET_TIME_S: float = 10.0
DEFAULT_DATASET_ROOT: Path = Path("data")
DEFAULT_PLAY_SOUNDS: bool = True


# Task definitions
class Task(str, Enum):
    CUBE = "cube"
    GBA = "gba"
    BALL = "ball"


class Setup(str, Enum):
    PHONE_TELEOP = "phone_teleop"
    LEADER_TELEOP = "leader_teleop"
    HAND_GUIDED = "hand_guided"


TASK_DESCRIPTIONS = {
    Task.CUBE: "Pick up the cube and place it in the target location",
    Task.GBA: "Press the up arrow on the GBA",
    Task.BALL: "Throw the ping-pong ball into the basket",
}


@dataclass
class CollectConfig:
    """Configuration for data collection."""

    task: Task
    setup: Setup
    # Collection mode: either num_episodes (regular) or benchmark_time_s (benchmark mode)
    num_episodes: int | None = DEFAULT_NUM_EPISODES
    benchmark_time_s: float | None = None  # If set, enables benchmark mode
    buffer_size: int = DEFAULT_BUFFER_SIZE
    fps: int = DEFAULT_FPS
    episode_time_s: float = DEFAULT_EPISODE_TIME_S
    reset_time_s: float = DEFAULT_RESET_TIME_S

    @property
    def is_benchmark_mode(self) -> bool:
        """True if running in benchmark mode (time-capped, RAM-only)."""
        return self.benchmark_time_s is not None

    # Robot config (from setup.py)
    robot_port: str = ROBOT_PORT
    robot_id: str = ROBOT_ID

    # Leader config (from setup.py)
    leader_port: str = LEADER_PORT
    leader_id: str = LEADER_ID

    # Camera config (from setup.py)
    wrist_camera_index: int = WRIST_CAMERA_INDEX
    top_camera_index: int | None = TOP_CAMERA_INDEX
    camera_width: int = CAMERA_WIDTH
    camera_height: int = CAMERA_HEIGHT

    # Dataset config (from setup.py)
    repo_id: str = HF_REPO_ID
    dataset_root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT)

    # Sound feedback
    play_sounds: bool = DEFAULT_PLAY_SOUNDS


@dataclass
class EpisodeFrame:
    """A single frame of data."""

    observation_state: dict[str, float]
    observation_images: dict[str, np.ndarray]
    action: dict[str, float]
    timestamp: float


@dataclass
class EpisodeBuffer:
    """Buffer for a single episode."""

    frames: list[EpisodeFrame] = field(default_factory=list)
    task: str = ""

    def add_frame(self, frame: EpisodeFrame) -> None:
        self.frames.append(frame)

    def clear(self) -> None:
        self.frames.clear()

    def __len__(self) -> int:
        return len(self.frames)


class MemoryBuffer:
    """
    Buffers multiple episodes in memory before flushing to disk.

    This prevents dataset corruption if the program crashes mid-recording.
    Episodes are only written to disk in batches.
    """

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.episodes: list[EpisodeBuffer] = []
        self.current_episode: EpisodeBuffer | None = None

    def start_episode(self, task: str) -> None:
        """Start recording a new episode."""
        self.current_episode = EpisodeBuffer(task=task)

    def add_frame(self, frame: EpisodeFrame) -> None:
        """Add a frame to the current episode."""
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode first.")
        self.current_episode.add_frame(frame)

    def save_episode(self) -> None:
        """Save the current episode to the buffer."""
        if self.current_episode is None:
            raise RuntimeError("No episode to save.")
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
        self.current_episode = None

    def discard_episode(self) -> None:
        """Discard the current episode (e.g., for re-recording)."""
        self.current_episode = None

    def should_flush(self) -> bool:
        """Check if buffer should be flushed to disk."""
        return len(self.episodes) >= self.buffer_size

    def get_episodes_to_flush(self) -> list[EpisodeBuffer]:
        """Get all buffered episodes and clear the buffer."""
        episodes = self.episodes
        self.episodes = []
        return episodes

    @property
    def num_buffered(self) -> int:
        return len(self.episodes)


def create_cameras(config: CollectConfig) -> dict[str, OpenCVCamera]:
    """Create camera instances based on config."""
    cameras = {}

    # Wrist camera (always present)
    wrist_config = OpenCVCameraConfig(
        index_or_path=config.wrist_camera_index,
        fps=config.fps,
        width=config.camera_width,
        height=config.camera_height,
    )
    cameras["wrist"] = OpenCVCamera(wrist_config)

    # Top camera (optional)
    if config.top_camera_index is not None:
        top_config = OpenCVCameraConfig(
            index_or_path=config.top_camera_index,
            fps=config.fps,
            width=config.camera_width,
            height=config.camera_height,
        )
        cameras["top"] = OpenCVCamera(top_config)

    return cameras


def create_hand_guided_arm(config: CollectConfig) -> SO101Leader:
    """
    Create arm for hand-guided mode.

    Uses SO101Leader since it has torque disabled by default,
    allowing free movement for hand-guided demonstration.
    """
    leader_config = SO101LeaderConfig(
        port=config.leader_port,  # Use leader port for hand-guided mode
        id=config.leader_id,
        use_degrees=True,
    )
    return SO101Leader(leader_config)


def create_leader_teleop_setup(
    config: CollectConfig,
) -> tuple[SO101Follower, SO101Leader]:
    """Create robot and leader for standard leader teleoperation."""
    # Follower (executes actions)
    follower_config = SO101FollowerConfig(
        port=config.robot_port,
        id=config.robot_id,
        use_degrees=True,
        cameras={},  # Cameras handled separately
    )
    follower = SO101Follower(follower_config)

    # Leader (provides actions)
    leader_config = SO101LeaderConfig(
        port=config.leader_port,
        id=config.leader_id,
        use_degrees=True,
    )
    leader = SO101Leader(leader_config)

    return follower, leader


def get_observation_features(cameras: dict[str, OpenCVCamera]) -> dict[str, Any]:
    """Get observation features for dataset creation."""
    # Single observation.state array with all 6 joint positions
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
        }
    }

    # Camera images
    for cam_name, cam in cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam.config.height, cam.config.width, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def get_action_features() -> dict[str, Any]:
    """Get action features for dataset creation."""
    # Single action array with all 6 joint positions
    return {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
        }
    }


def build_dataset_features(cameras: dict[str, OpenCVCamera]) -> dict[str, Any]:
    """Build complete feature dict for LeRobotDataset."""
    features = {}
    features.update(get_observation_features(cameras))
    features.update(get_action_features())
    return features


def record_loop_hand_guided(
    arm: SO101Leader,
    cameras: dict[str, OpenCVCamera],
    memory_buffer: MemoryBuffer,
    events: dict,
    config: CollectConfig,
    task_description: str,
) -> None:
    """
    Recording loop for hand-guided mode.

    In this mode, the same arm provides both observation.state and action.
    """
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < config.episode_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Read arm position (used as both observation and action)
        arm_state = arm.get_action()  # Returns {motor.pos: value}

        # Read cameras
        images = {}
        for cam_name, cam in cameras.items():
            try:
                image = cam.async_read()
                if image is not None:
                    images[cam_name] = image
            except (TimeoutError, Exception) as e:
                # Log timeout/errors but continue with other cameras
                logger.warning(f"Camera {cam_name} read failed: {e}")
                continue

        # Create frame with identical observation.state and action
        frame = EpisodeFrame(
            observation_state=arm_state,
            observation_images=images,
            action=arm_state.copy(),  # Same as observation for hand-guided
            timestamp=timestamp,
        )
        memory_buffer.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / config.fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


def record_loop_leader_teleop(
    robot: SO101Follower,
    leader: SO101Leader,
    cameras: dict[str, OpenCVCamera],
    memory_buffer: MemoryBuffer,
    events: dict,
    config: CollectConfig,
    task_description: str,
) -> None:
    """
    Recording loop for leader teleoperation mode.

    Leader arm provides action, follower arm provides observation.state.
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
                # Log timeout/errors but continue with other cameras
                logger.warning(f"Camera {cam_name} read failed: {e}")
                continue

        frame = EpisodeFrame(
            observation_state=obs_state,
            observation_images=images,
            action=action,
            timestamp=timestamp,
        )
        memory_buffer.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / config.fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


def reset_loop(
    events: dict,
    config: CollectConfig,
) -> None:
    """Wait for environment reset between episodes."""
    timestamp = 0.0
    start_t = time.perf_counter()

    while timestamp < config.reset_time_s:
        if events["exit_early"]:
            events["exit_early"] = False
            break

        precise_sleep(0.1)
        timestamp = time.perf_counter() - start_t


def check_and_remove_existing_dataset(dataset_path: Path) -> None:
    """
    Check if dataset directory exists and prompt user to delete it.

    Raises SystemExit if user declines to delete.
    """
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


def _state_dict_to_array(state_dict: dict[str, float]) -> np.ndarray:
    """
    Convert a state dict like {shoulder_pan.pos: value, ...} to a numpy array.

    The array order matches MOTOR_NAMES.
    """
    values = []
    for motor_name in MOTOR_NAMES:
        key = f"{motor_name}.pos"
        if key in state_dict:
            values.append(state_dict[key])
        else:
            raise KeyError(
                f"Missing motor key {key} in state dict. Available: {list(state_dict.keys())}"
            )
    return np.array(values, dtype=np.float32)


def flush_to_dataset(
    episodes: list[EpisodeBuffer],
    dataset: LeRobotDataset,
    config: CollectConfig,
) -> float:
    """
    Flush buffered episodes to LeRobotDataset.

    Returns the time spent encoding (for benchmark tracking).
    """
    start_time = time.perf_counter()

    for episode in episodes:
        for i, frame in enumerate(episode.frames):
            # Build frame dict for LeRobotDataset
            frame_dict: dict[str, Any] = {"task": episode.task}

            # Convert observation state dict to array
            frame_dict["observation.state"] = _state_dict_to_array(
                frame.observation_state
            )

            # Add observation images
            for cam_name, image in frame.observation_images.items():
                frame_dict[f"observation.images.{cam_name}"] = image

            # Convert action dict to array
            frame_dict["action"] = _state_dict_to_array(frame.action)

            dataset.add_frame(frame_dict)

        dataset.save_episode()

    encoding_time = time.perf_counter() - start_time
    return encoding_time


def collect(config: CollectConfig) -> None:
    """Main data collection function."""
    init_logging()

    logger.info(
        f"Starting data collection: task={config.task.value}, setup={config.setup.value}"
    )
    if config.is_benchmark_mode:
        logger.info(
            f"BENCHMARK MODE: {config.benchmark_time_s / 60:.1f} minutes, RAM-only (no encoding during collection)"
        )
    else:
        logger.info(f"Regular mode: {config.num_episodes} episodes")
        logger.info(f"Buffer size: {config.buffer_size}")

    task_description = TASK_DESCRIPTIONS[config.task]

    # Initialize benchmark tracker
    tracker = BenchmarkTracker()
    session_start = time.time()
    total_encoding_time = 0.0
    mistakes = 0
    total_frames = 0

    # Create cameras
    cameras = create_cameras(config)

    # Create memory buffer
    memory_buffer = MemoryBuffer(buffer_size=config.buffer_size)

    # Create dataset
    dataset_name = f"{config.task.value}_{config.setup.value}"
    # Enforce naming convention: HuggingFace repo names should use underscores, not hyphens
    assert "-" not in config.repo_id, (
        f"repo_id '{config.repo_id}' contains hyphens. "
        "Use underscores instead (e.g., 'user_name/repo_name')."
    )
    repo_id = f"{config.repo_id}_{dataset_name}"
    dataset_path = config.dataset_root / dataset_name

    # Check if dataset directory exists and prompt for deletion
    check_and_remove_existing_dataset(dataset_path)

    # Build features from observation/action structure
    features = build_dataset_features(cameras)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.fps,
        root=dataset_path,
        robot_type="so101",
        features=features,
        use_videos=True,
    )

    # Setup based on collection setup
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
        elif config.setup == Setup.PHONE_TELEOP:
            raise NotImplementedError("Phone teleop not yet implemented")

        # Connect cameras
        for cam in cameras.values():
            cam.connect()

        # Setup keyboard listener
        listener, events = init_keyboard_listener()

        recorded_episodes = 0
        collection_start = time.perf_counter()
        session_elapsed = 0.0

        # Determine loop condition based on mode
        def should_continue() -> bool:
            if config.is_benchmark_mode:
                return (
                    session_elapsed < config.benchmark_time_s
                    and not events["stop_recording"]
                )
            else:
                return (
                    recorded_episodes < config.num_episodes
                    and not events["stop_recording"]
                )

        while should_continue():
            episode_num = recorded_episodes + 1
            if config.is_benchmark_mode:
                remaining_min = (config.benchmark_time_s - session_elapsed) / 60
                log_say(
                    f"Recording episode {episode_num} ({remaining_min:.1f} min remaining)",
                    config.play_sounds,
                )
            else:
                log_say(
                    f"Recording episode {episode_num} of {config.num_episodes}",
                    config.play_sounds,
                )

            # Start new episode
            memory_buffer.start_episode(task_description)

            # Record based on setup
            if config.setup == Setup.HAND_GUIDED:
                record_loop_hand_guided(
                    arm=arm,
                    cameras=cameras,
                    memory_buffer=memory_buffer,
                    events=events,
                    config=config,
                    task_description=task_description,
                )
            elif config.setup == Setup.LEADER_TELEOP:
                record_loop_leader_teleop(
                    robot=robot,
                    leader=leader,
                    cameras=cameras,
                    memory_buffer=memory_buffer,
                    events=events,
                    config=config,
                    task_description=task_description,
                )

            # Handle re-record
            if events["rerecord_episode"]:
                log_say("Re-recording episode", config.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                memory_buffer.discard_episode()
                mistakes += 1
                continue

            # Save episode to buffer
            if memory_buffer.current_episode:
                total_frames += len(memory_buffer.current_episode)
            memory_buffer.save_episode()
            recorded_episodes += 1

            # Flush to disk if buffer is full (skip in benchmark mode - keep everything in RAM)
            if not config.is_benchmark_mode and memory_buffer.should_flush():
                log_say("Saving to disk...", config.play_sounds)
                episodes_to_flush = memory_buffer.get_episodes_to_flush()
                encoding_time = flush_to_dataset(episodes_to_flush, dataset, config)
                total_encoding_time += encoding_time
                logger.info(
                    f"Flushed {len(episodes_to_flush)} episodes to disk "
                    f"(encoding took {encoding_time:.1f}s)"
                )

            # Update elapsed time (excluding encoding time)
            session_elapsed = (
                time.perf_counter() - collection_start - total_encoding_time
            )

            # Reset time (skip if done)
            if should_continue():
                log_say("Reset the environment", config.play_sounds)
                reset_loop(events, config)
                session_elapsed = (
                    time.perf_counter() - collection_start - total_encoding_time
                )

        collection_end = time.perf_counter()
        total_collection_time = collection_end - collection_start

        # Record metrics BEFORE encoding (so benchmark metrics reflect pure collection time)
        session_end = time.time()
        metrics = SessionMetrics(
            task=config.task.value,
            setup=config.setup.value,
            session_start=session_start,
            session_end=session_end,
            collection_time_s=total_collection_time - total_encoding_time,
            encoding_time_s=total_encoding_time,  # Will be 0 for benchmark mode at this point
            episodes_recorded=recorded_episodes,
            total_frames=total_frames,
            mistakes=mistakes,
        )
        tracker.log_session(metrics)
        logger.info(f"Session metrics saved to {tracker.csv_path}")

        # Flush all remaining episodes to disk
        if memory_buffer.num_buffered > 0:
            if config.is_benchmark_mode:
                log_say(
                    f"Encoding {memory_buffer.num_buffered} episodes to disk...",
                    config.play_sounds,
                )
            else:
                log_say("Saving remaining episodes...", config.play_sounds)
            episodes_to_flush = memory_buffer.get_episodes_to_flush()
            encoding_time = flush_to_dataset(episodes_to_flush, dataset, config)
            total_encoding_time += encoding_time
            logger.info(f"Encoding took {encoding_time:.1f}s")

        # Finalize dataset
        dataset.finalize()

        log_say("Recording complete", config.play_sounds, blocking=True)
        logger.info(f"Recorded {recorded_episodes} episodes")
        logger.info(f"Total collection time: {total_collection_time:.1f}s")
        logger.info(f"Total encoding time: {total_encoding_time:.1f}s")
        logger.info(
            f"Net collection time: {total_collection_time - total_encoding_time:.1f}s"
        )

        # Prompt to push to HuggingFace Hub
        push_response = input(f"\nPush dataset to HuggingFace Hub ({repo_id})? [y/N]: ")
        push_response_clean = push_response.strip().lower()
        logger.info(
            f"Push response received: {repr(push_response)} "
            f"(cleaned: {repr(push_response_clean)}, "
            f"matches: {push_response_clean in ('y', 'yes')})"
        )
        if push_response_clean in ("y", "yes"):
            logger.info(f"Pushing dataset to {repo_id}...")
            try:
                dataset.push_to_hub()
                logger.info("Dataset pushed successfully!")
            except Exception as e:
                logger.error(f"Failed to push dataset to Hub: {e}", exc_info=True)
                logger.info("Dataset saved locally.")
        else:
            logger.info("Skipping push to Hub. Dataset saved locally.")

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


def parse_args() -> CollectConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SO-101 Benchmark Data Collection",
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

    # Collection mode: either --num-episodes (regular) or --benchmark (time-capped)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to record (regular mode)",
    )
    mode_group.add_argument(
        "--benchmark",
        type=float,
        metavar="SECONDS",
        help="Benchmark mode: collect for N seconds with no encoding (default: 900 = 15 min). "
        "All data stays in RAM during collection.",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help="Number of episodes to buffer before flushing to disk (regular mode only)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Recording FPS",
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=DEFAULT_EPISODE_TIME_S,
        help="Maximum episode duration in seconds",
    )
    parser.add_argument(
        "--reset-time",
        type=float,
        default=DEFAULT_RESET_TIME_S,
        help="Time for environment reset between episodes",
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
        help="Leader arm USB port (for leader_teleop mode)",
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
        default=HF_REPO_ID,
        help="Base HuggingFace repo ID",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable sound feedback",
    )

    args = parser.parse_args()

    # Determine mode: benchmark or regular
    if args.benchmark is not None:
        num_episodes = None
        benchmark_time_s = args.benchmark
    else:
        num_episodes = args.num_episodes
        benchmark_time_s = None

    return CollectConfig(
        task=Task(args.task),
        setup=Setup(args.setup),
        num_episodes=num_episodes,
        benchmark_time_s=benchmark_time_s,
        buffer_size=args.buffer_size,
        fps=args.fps,
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
        play_sounds=not args.no_sound,
    )


def main() -> None:
    config = parse_args()
    collect(config)


if __name__ == "__main__":
    main()
