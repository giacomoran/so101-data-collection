#!/usr/bin/env python
"""
Data collection script for SO-101 benchmark.

Supports three collection methods:
- phone_teleop: Phone controls end-effector
- leader_teleop: Standard leader-follower teleoperation
- hand_guided: Single arm provides both observation and action

Usage:
    python -m src.collect \
        --task pick_place_cube \
        --method hand_guided \
        --num-episodes 50 \
        --buffer-size 10

    # Warmup mode (no saving):
    python -m src.collect \
        --task pick_place_cube \
        --method hand_guided \
        --warmup \
        --num-episodes 10
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

# Default collection parameters
DEFAULT_NUM_EPISODES: int = 50
DEFAULT_BUFFER_SIZE: int = 10
DEFAULT_FPS: int = 30
DEFAULT_EPISODE_TIME_S: float = 20.0
DEFAULT_RESET_TIME_S: float = 10.0
DEFAULT_DATASET_ROOT: Path = Path("data")
DEFAULT_PLAY_SOUNDS: bool = True


# Task definitions
class Task(str, Enum):
    PICK_PLACE_CUBE = "pick_place_cube"
    PRESS_GBA = "press_gba"
    THROW_BALL = "throw_ball"


class Method(str, Enum):
    PHONE_TELEOP = "phone_teleop"
    LEADER_TELEOP = "leader_teleop"
    HAND_GUIDED = "hand_guided"


TASK_DESCRIPTIONS = {
    Task.PICK_PLACE_CUBE: "Pick up the cube and place it in the target location",
    Task.PRESS_GBA: "Press the up arrow on the GBA",
    Task.THROW_BALL: "Throw the ping-pong ball into the basket",
}


@dataclass
class CollectConfig:
    """Configuration for data collection."""

    task: Task
    method: Method
    num_episodes: int = DEFAULT_NUM_EPISODES
    warmup: bool = False
    buffer_size: int = DEFAULT_BUFFER_SIZE
    fps: int = DEFAULT_FPS
    episode_time_s: float = DEFAULT_EPISODE_TIME_S
    reset_time_s: float = DEFAULT_RESET_TIME_S

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
    # Motor positions (6 joints)
    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    features = {
        f"observation.state.{name}.pos": {"dtype": "float32", "shape": (1,)}
        for name in motor_names
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
    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    return {
        f"action.{name}.pos": {"dtype": "float32", "shape": (1,)}
        for name in motor_names
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

            # Add observation state
            for motor_key, value in frame.observation_state.items():
                frame_dict[f"observation.state.{motor_key}"] = np.array(
                    [value], dtype=np.float32
                )

            # Add observation images
            for cam_name, image in frame.observation_images.items():
                frame_dict[f"observation.images.{cam_name}"] = image

            # Add action
            for motor_key, value in frame.action.items():
                frame_dict[f"action.{motor_key}"] = np.array([value], dtype=np.float32)

            dataset.add_frame(frame_dict)

        dataset.save_episode()

    encoding_time = time.perf_counter() - start_time
    return encoding_time


def collect(config: CollectConfig) -> None:
    """Main data collection function."""
    init_logging()

    logger.info(
        f"Starting data collection: task={config.task.value}, method={config.method.value}"
    )
    logger.info(f"Warmup mode: {config.warmup}")
    logger.info(f"Episodes to record: {config.num_episodes}")
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

    # Create dataset (only if not in warmup mode)
    dataset = None
    if not config.warmup:
        dataset_name = f"{config.task.value}_{config.method.value}"
        repo_id = f"{config.repo_id}-{dataset_name}"
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

    # Setup based on method
    arm = None
    robot = None
    leader = None
    listener = None

    try:
        if config.method == Method.HAND_GUIDED:
            arm = create_hand_guided_arm(config)
            arm.connect()
        elif config.method == Method.LEADER_TELEOP:
            robot, leader = create_leader_teleop_setup(config)
            robot.connect()
            leader.connect()
        elif config.method == Method.PHONE_TELEOP:
            raise NotImplementedError("Phone teleop not yet implemented")

        # Connect cameras
        for cam in cameras.values():
            cam.connect()

        # Setup keyboard listener
        listener, events = init_keyboard_listener()

        recorded_episodes = 0
        collection_start = time.perf_counter()

        while recorded_episodes < config.num_episodes and not events["stop_recording"]:
            episode_num = recorded_episodes + 1
            log_say(
                f"Recording episode {episode_num} of {config.num_episodes}",
                config.play_sounds,
            )

            # Start new episode
            memory_buffer.start_episode(task_description)

            # Record based on method
            if config.method == Method.HAND_GUIDED:
                record_loop_hand_guided(
                    arm=arm,
                    cameras=cameras,
                    memory_buffer=memory_buffer,
                    events=events,
                    config=config,
                    task_description=task_description,
                )
            elif config.method == Method.LEADER_TELEOP:
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

            # Flush to disk if buffer is full (and not in warmup mode)
            if memory_buffer.should_flush() and dataset is not None:
                log_say("Saving to disk...", config.play_sounds)
                episodes_to_flush = memory_buffer.get_episodes_to_flush()
                encoding_time = flush_to_dataset(episodes_to_flush, dataset, config)
                total_encoding_time += encoding_time
                logger.info(
                    f"Flushed {len(episodes_to_flush)} episodes to disk "
                    f"(encoding took {encoding_time:.1f}s)"
                )

            # Reset time (skip for last episode)
            if recorded_episodes < config.num_episodes and not events["stop_recording"]:
                log_say("Reset the environment", config.play_sounds)
                reset_loop(events, config)

        collection_end = time.perf_counter()

        # Flush remaining episodes
        if memory_buffer.num_buffered > 0 and dataset is not None:
            log_say("Saving remaining episodes...", config.play_sounds)
            episodes_to_flush = memory_buffer.get_episodes_to_flush()
            encoding_time = flush_to_dataset(episodes_to_flush, dataset, config)
            total_encoding_time += encoding_time

        # Finalize dataset
        if dataset is not None:
            dataset.finalize()

        session_end = time.time()
        total_collection_time = collection_end - collection_start

        # Record metrics
        if not config.warmup:
            metrics = SessionMetrics(
                task=config.task.value,
                method=config.method.value,
                session_start=session_start,
                session_end=session_end,
                collection_time_s=total_collection_time - total_encoding_time,
                encoding_time_s=total_encoding_time,
                episodes_recorded=recorded_episodes,
                total_frames=total_frames,
                mistakes=mistakes,
            )
            tracker.log_session(metrics)
            logger.info(f"Session metrics saved to {tracker.csv_path}")

        log_say("Recording complete", config.play_sounds, blocking=True)
        logger.info(f"Recorded {recorded_episodes} episodes")
        logger.info(f"Total collection time: {total_collection_time:.1f}s")
        logger.info(f"Total encoding time: {total_encoding_time:.1f}s")
        logger.info(
            f"Net collection time: {total_collection_time - total_encoding_time:.1f}s"
        )

        # Prompt to push to HuggingFace Hub
        if dataset is not None:
            dataset_name = f"{config.task.value}_{config.method.value}"
            repo_id = f"{config.repo_id}-{dataset_name}"
            push_response = input(
                f"\nPush dataset to HuggingFace Hub ({repo_id})? [y/N]: "
            )
            push_response_clean = push_response.strip().lower()
            logger.info(
                f"Push response received: {repr(push_response)} "
                f"(cleaned: {repr(push_response_clean)}, "
                f"matches: {push_response_clean in ('y', 'yes')})"
            )
            if push_response_clean in ("y", "yes"):
                logger.info(f"Pushing dataset to {repo_id}...")
                try:
                    dataset.push_to_hub(repo_id=repo_id)
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
        "--method",
        type=str,
        required=True,
        choices=[m.value for m in Method],
        help="Data collection method",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warmup mode: run demos without saving to dataset",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help="Number of episodes to buffer before flushing to disk",
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

    return CollectConfig(
        task=Task(args.task),
        method=Method(args.method),
        num_episodes=args.num_episodes,
        warmup=args.warmup,
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
