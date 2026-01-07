#!/usr/bin/env python
"""
WARN: I did not reach a satisfactory conclusion with this test for data collection. See
outputs/verify_alignment for the results of verify_alignment.py and
verify_alignment_qr.py on ad-hoc episodes. Use this file at your own risk.


Data collection with offline frame alignment for SO-101.

Strategy:
1. Camera capture thread: flush buffer on every read, keep only newest frame
   - Records received timestamp for each frame
   - Subtracts known camera latency to get actual capture time
2. Proprioception thread: sample at high rate (100-200Hz), timestamp at read
3. Store raw streams independently during collection
4. Align frames OFFLINE:
   - Output dataset FPS is configurable (10Hz or 20Hz typical)
   - Pick most recent camera frame for each timestep
   - Linearly interpolate proprioception at camera frame timestamp

This approach decouples capture from alignment for more accurate temporal alignment.

For vanilla collection without alignment, use collect.py.

Usage:
    # Basic aligned collection at 10Hz output
    python -m so101_data_collection.collect.collect_aligned \
        --task cube \
        --setup hand_guided \
        --num-episodes 10 \
        --dataset-fps 10

    # With custom proprioception rate and camera latency
    python -m so101_data_collection.collect.collect_aligned \
        --task cube \
        --setup hand_guided \
        --num-episodes 10 \
        --proprioception-fps 200 \
        --camera-latency-ms 20

    # With explicit repo ID
    python -m so101_data_collection.collect.collect_aligned \
        --task cube \
        --setup hand_guided \
        --repo-id giacomoran/my_aligned_dataset \
        --num-episodes 10
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
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
DEFAULT_DATASET_FPS = 10  # Output dataset FPS
DEFAULT_PROPRIOCEPTION_FPS = 100  # Hz for proprioception sampling
DEFAULT_CAMERA_LATENCY_MS = 20.0  # Known camera latency to subtract
DEFAULT_EPISODE_TIME_S = 20.0
DEFAULT_RESET_TIME_S = 10.0
DEFAULT_DATASET_ROOT = Path("data")


class Task(str, Enum):
    CUBE = "cube"
    GBA = "gba"
    BALL = "ball"


class Setup(str, Enum):
    HAND_GUIDED = "hand_guided"
    LEADER_TELEOP = "leader_teleop"


TASK_DESCRIPTIONS = {
    Task.CUBE: "Pick up the cube and place it in the target location",
    Task.GBA: "Press the up arrow on the GBA",
    Task.BALL: "Throw the ping-pong ball into the basket",
}


@dataclass
class CollectAlignedConfig:
    """Configuration for aligned data collection."""

    task: Task
    setup: Setup
    num_episodes: int = DEFAULT_NUM_EPISODES
    dataset_fps: int = DEFAULT_DATASET_FPS
    proprioception_fps: int = DEFAULT_PROPRIOCEPTION_FPS
    camera_latency_ms: float = DEFAULT_CAMERA_LATENCY_MS
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
# Raw Data Structures
# ============================================================================


@dataclass
class TimestampedImage:
    """A camera frame with its estimated actual capture time."""

    image: np.ndarray
    capture_time: float  # Estimated time when light hit sensor (receive_time - latency)
    receive_time: float  # Time when we got the frame


@dataclass
class TimestampedProprio:
    """A proprioception reading with its capture time."""

    data: dict[str, float]
    capture_time: float


@dataclass
class RawEpisodeData:
    """Raw captured data for one episode, before alignment."""

    camera_frames: dict[str, list[TimestampedImage]] = field(default_factory=dict)
    proprio_readings: list[TimestampedProprio] = field(default_factory=list)
    task: str = ""
    episode_start_time: float = 0.0


# ============================================================================
# Camera Capture Thread
# ============================================================================


class FlushingCameraCapture:
    """
    Camera capture with buffer flushing to minimize latency.

    Runs in a background thread, continuously capturing frames and flushing
    the buffer. The main thread can get the most recent frame at any time.
    """

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        latency_ms: float,
        name: str = "camera",
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.latency_s = latency_ms / 1000.0
        self.name = name

        self.cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: TimestampedImage | None = None
        self._frame_count = 0

    def connect(self) -> None:
        """Open camera and start capture thread."""
        # Try AVFoundation backend first (macOS), then any available
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap.isOpened():
                logger.info(f"{self.name}: Opened with backend {backend}")
                break
            self.cap.release()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request native 30 FPS

        # Log actual settings
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"{self.name}: {actual_w}x{actual_h} @ {actual_fps}fps")

        # Warm up camera
        logger.info(f"{self.name}: Warming up...")
        for _ in range(30):
            self.cap.read()

        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"{self.name}: Capture thread started")

    def _flush_and_read(self) -> tuple[np.ndarray | None, float]:
        """Flush buffer and read the newest frame."""
        # Flush buffer by grabbing (but not decoding) a few frames
        for _ in range(3):
            self.cap.grab()

        # Now read (decode) the latest frame
        receive_time = time.perf_counter()
        ret, frame = self.cap.read()

        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame, receive_time
        return None, receive_time

    def _capture_loop(self) -> None:
        """Background thread that continuously captures frames."""
        while self._running:
            frame, receive_time = self._flush_and_read()

            if frame is not None:
                # Estimate actual capture time by subtracting known latency
                capture_time = receive_time - self.latency_s

                with self._lock:
                    self._latest_frame = TimestampedImage(
                        image=frame,
                        capture_time=capture_time,
                        receive_time=receive_time,
                    )
                    self._frame_count += 1

            # Small sleep to prevent CPU spinning (camera is ~30fps anyway)
            time.sleep(0.001)

    def get_latest(self) -> TimestampedImage | None:
        """Get the most recent frame (thread-safe)."""
        with self._lock:
            return self._latest_frame

    def disconnect(self) -> None:
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        logger.info(f"{self.name}: Disconnected (captured {self._frame_count} frames)")


# ============================================================================
# Proprioception Sampling Thread
# ============================================================================


class ProprioceptionSampler:
    """
    High-rate proprioception sampler.

    Runs in a background thread, sampling joint positions at the configured rate.
    All readings are stored with timestamps for offline alignment.
    """

    def __init__(self, arm: SO101Leader | SO101Follower, rate_hz: int):
        self.arm = arm
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._readings: list[TimestampedProprio] = []
        self._reading_count = 0

    def start(self) -> None:
        """Start the sampling thread."""
        self._running = True
        self._readings = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(f"Proprioception sampler started at {self.rate_hz}Hz")

    def _sample_loop(self) -> None:
        """Background thread that samples proprioception."""
        while self._running:
            loop_start = time.perf_counter()

            # Read proprioception
            capture_time = time.perf_counter()
            if isinstance(self.arm, SO101Leader):
                data = self.arm.get_action()
            else:
                obs = self.arm.get_observation()
                data = {k: v for k, v in obs.items() if k.endswith(".pos")}

            with self._lock:
                self._readings.append(
                    TimestampedProprio(data=data.copy(), capture_time=capture_time)
                )
                self._reading_count += 1

            # Sleep to maintain rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.period_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> list[TimestampedProprio]:
        """Stop sampling and return all readings."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        with self._lock:
            readings = self._readings.copy()
            self._readings = []

        logger.info(f"Proprioception sampler stopped ({len(readings)} readings)")
        return readings

    def get_readings_since(self, start_time: float) -> list[TimestampedProprio]:
        """Get a copy of readings since start_time (for monitoring, not final use)."""
        with self._lock:
            return [r for r in self._readings if r.capture_time >= start_time]


# ============================================================================
# Offline Alignment
# ============================================================================


def interpolate_proprio(
    readings: list[TimestampedProprio],
    target_time: float,
) -> dict[str, float] | None:
    """
    Linearly interpolate proprioception at a target time.

    Returns None if target_time is outside the readings range.
    """
    if not readings:
        return None

    # Find bracketing readings
    before: TimestampedProprio | None = None
    after: TimestampedProprio | None = None

    for reading in readings:
        if reading.capture_time <= target_time:
            before = reading
        elif after is None:
            after = reading
            break

    if before is None:
        return None  # Target time is before all readings

    if after is None:
        # Target time is after all readings, use last reading
        return before.data.copy()

    # Linear interpolation
    t0 = before.capture_time
    t1 = after.capture_time
    alpha = (target_time - t0) / (t1 - t0) if t1 > t0 else 0.0

    interpolated = {}
    for key in before.data:
        v0 = before.data[key]
        v1 = after.data.get(key, v0)
        interpolated[key] = v0 + alpha * (v1 - v0)

    return interpolated


def align_episode(
    raw_data: RawEpisodeData,
    dataset_fps: int,
    episode_duration_s: float,
) -> list[dict[str, Any]]:
    """
    Align raw episode data to fixed-rate output frames.

    For each output timestep:
    - Pick the most recent camera frame
    - Linearly interpolate proprioception at the camera frame's capture time

    Returns list of aligned frame dicts ready for dataset.
    """
    aligned_frames = []
    output_period = 1.0 / dataset_fps
    start_time = raw_data.episode_start_time

    # Sort proprioception readings by time
    proprio = sorted(raw_data.proprio_readings, key=lambda r: r.capture_time)

    # Get all camera names
    camera_names = list(raw_data.camera_frames.keys())

    # Generate output timesteps
    num_frames = int(episode_duration_s * dataset_fps)

    for frame_idx in range(num_frames):
        output_time = start_time + frame_idx * output_period

        frame_dict: dict[str, Any] = {"task": raw_data.task}

        # For each camera, find the most recent frame before output_time
        camera_capture_time = None
        for cam_name in camera_names:
            frames = raw_data.camera_frames.get(cam_name, [])
            if not frames:
                continue

            # Find most recent frame
            best_frame = None
            for f in frames:
                if f.capture_time <= output_time:
                    best_frame = f
                else:
                    break

            if best_frame is not None:
                frame_dict[f"observation.images.{cam_name}"] = best_frame.image
                # Use wrist camera time as reference for proprioception alignment
                if cam_name == "wrist" or camera_capture_time is None:
                    camera_capture_time = best_frame.capture_time

        # Skip frame if no camera data
        if camera_capture_time is None:
            logger.warning(f"No camera frame for output timestep {frame_idx}")
            continue

        # Interpolate proprioception at camera capture time
        proprio_data = interpolate_proprio(proprio, camera_capture_time)
        if proprio_data is None:
            logger.warning(
                f"No proprioception data for camera time at frame {frame_idx}"
            )
            continue

        # Convert proprioception dict to arrays
        state_array = _state_dict_to_array(proprio_data)
        frame_dict["observation.state"] = state_array
        frame_dict["action"] = state_array.copy()  # Same for hand-guided

        aligned_frames.append(frame_dict)

    logger.info(
        f"Aligned {len(aligned_frames)} frames from "
        f"{sum(len(f) for f in raw_data.camera_frames.values())} camera frames "
        f"and {len(proprio)} proprioception readings"
    )

    return aligned_frames


def _state_dict_to_array(state_dict: dict[str, float]) -> np.ndarray:
    """Convert state dict to numpy array in MOTOR_NAMES order."""
    values = []
    for motor_name in MOTOR_NAMES:
        key = f"{motor_name}.pos"
        if key in state_dict:
            values.append(state_dict[key])
        else:
            raise KeyError(f"Missing motor key {key}")
    return np.array(values, dtype=np.float32)


# ============================================================================
# Hardware Setup
# ============================================================================


def create_arm(config: CollectAlignedConfig) -> SO101Leader:
    """Create arm for hand-guided mode."""
    leader_config = SO101LeaderConfig(
        port=config.leader_port,
        id=config.leader_id,
        use_degrees=True,
    )
    return SO101Leader(leader_config)


def create_cameras(
    config: CollectAlignedConfig,
) -> dict[str, FlushingCameraCapture]:
    """Create camera instances with flushing capture."""
    cameras = {}

    cameras["wrist"] = FlushingCameraCapture(
        camera_index=config.wrist_camera_index,
        width=config.camera_width,
        height=config.camera_height,
        latency_ms=config.camera_latency_ms,
        name="wrist",
    )

    if config.top_camera_index is not None:
        cameras["top"] = FlushingCameraCapture(
            camera_index=config.top_camera_index,
            width=config.camera_width,
            height=config.camera_height,
            latency_ms=config.camera_latency_ms,
            name="top",
        )

    return cameras


# ============================================================================
# Dataset Features
# ============================================================================


def get_observation_features(
    cameras: dict[str, FlushingCameraCapture],
) -> dict[str, Any]:
    """Get observation features for dataset creation.

    IMPORTANT: State names must include '.pos' suffix to match the format
    used by lerobot-record (SO101Follower.observation_features uses '.pos').
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{name}.pos" for name in MOTOR_NAMES],
        }
    }

    for cam_name, cam in cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def get_action_features() -> dict[str, Any]:
    """Get action features for dataset creation.

    IMPORTANT: Action names must include '.pos' suffix so that lerobot-replay
    creates action dicts with keys like 'shoulder_pan.pos', which is what
    robot.send_action() expects (see so101_follower.py line 209).
    """
    return {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{name}.pos" for name in MOTOR_NAMES],
        }
    }


def build_dataset_features(
    cameras: dict[str, FlushingCameraCapture],
) -> dict[str, Any]:
    """Build complete feature dict for LeRobotDataset."""
    features = {}
    features.update(get_observation_features(cameras))
    features.update(get_action_features())
    return features


# ============================================================================
# Recording
# ============================================================================


def record_episode_raw(
    arm: SO101Leader,
    cameras: dict[str, FlushingCameraCapture],
    proprio_sampler: ProprioceptionSampler,
    events: dict,
    config: CollectAlignedConfig,
    task_description: str,
) -> RawEpisodeData:
    """
    Record a raw episode with async capture.

    Captures camera frames and proprioception independently, stores with timestamps.
    Alignment happens offline after recording.
    """
    raw_data = RawEpisodeData(task=task_description)

    # Initialize camera frame lists
    for cam_name in cameras:
        raw_data.camera_frames[cam_name] = []

    # Record episode start time
    raw_data.episode_start_time = time.perf_counter()

    # Start proprioception sampling
    proprio_sampler.start()

    # Track last captured frame to avoid duplicates
    last_frame_time: dict[str, float] = {cam: 0.0 for cam in cameras}

    timestamp = 0.0
    start_t = time.perf_counter()

    # Main capture loop - runs faster than output FPS to capture all camera frames
    capture_rate = 60  # Hz - capture loop rate
    capture_period = 1.0 / capture_rate

    while timestamp < config.episode_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Capture latest frame from each camera (if new)
        for cam_name, cam in cameras.items():
            frame = cam.get_latest()
            if frame is not None and frame.capture_time > last_frame_time[cam_name]:
                raw_data.camera_frames[cam_name].append(frame)
                last_frame_time[cam_name] = frame.capture_time

        # Update timestamp
        timestamp = time.perf_counter() - start_t

        # Sleep to maintain capture rate
        elapsed = time.perf_counter() - loop_start
        sleep_time = capture_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Stop proprioception sampling and get all readings
    raw_data.proprio_readings = proprio_sampler.stop()

    logger.info(
        f"Recorded episode: "
        f"{sum(len(f) for f in raw_data.camera_frames.values())} camera frames, "
        f"{len(raw_data.proprio_readings)} proprioception readings, "
        f"{timestamp:.1f}s duration"
    )

    return raw_data


def flush_to_dataset(
    aligned_frames: list[dict[str, Any]],
    dataset: LeRobotDataset,
) -> None:
    """Write aligned frames to dataset."""
    for frame_dict in aligned_frames:
        dataset.add_frame(frame_dict)
    dataset.save_episode()


# ============================================================================
# Dataset Management
# ============================================================================


def get_repo_id(config: CollectAlignedConfig) -> str:
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


def collect_aligned(config: CollectAlignedConfig) -> None:
    """Main aligned data collection function."""
    init_logging()

    repo_id = get_repo_id(config)
    task_description = TASK_DESCRIPTIONS[config.task]

    logger.info(
        f"Starting aligned collection: task={config.task.value}, setup={config.setup.value}"
    )
    logger.info(f"Dataset: {repo_id}")
    logger.info(
        f"Dataset FPS: {config.dataset_fps}Hz, Proprioception FPS: {config.proprioception_fps}Hz"
    )
    logger.info(f"Camera latency compensation: {config.camera_latency_ms}ms")

    # Create hardware
    cameras = create_cameras(config)
    arm = create_arm(config)

    # Create dataset
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

    listener = None

    try:
        # Connect hardware
        arm.connect()
        for cam in cameras.values():
            cam.connect()

        # Setup keyboard listener
        listener, events = init_keyboard_listener()

        # Create proprioception sampler (reused per episode)
        proprio_sampler = ProprioceptionSampler(arm, config.proprioception_fps)

        recorded_episodes = 0

        while recorded_episodes < config.num_episodes and not events["stop_recording"]:
            episode_num = recorded_episodes + 1
            log_say(
                f"Recording episode {episode_num} of {config.num_episodes}",
                config.play_sounds,
            )

            # Record raw episode
            raw_data = record_episode_raw(
                arm=arm,
                cameras=cameras,
                proprio_sampler=proprio_sampler,
                events=events,
                config=config,
                task_description=task_description,
            )

            # Handle re-record
            if events["rerecord_episode"]:
                log_say("Re-recording episode", config.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                continue

            # Align episode offline
            log_say("Aligning episode...", config.play_sounds)
            aligned_frames = align_episode(
                raw_data=raw_data,
                dataset_fps=config.dataset_fps,
                episode_duration_s=config.episode_time_s,
            )

            if len(aligned_frames) > 0:
                # Write to dataset
                flush_to_dataset(aligned_frames, dataset)
                recorded_episodes += 1
                logger.info(
                    f"Episode {episode_num} saved ({len(aligned_frames)} frames)"
                )
            else:
                logger.warning("Episode had no aligned frames, discarding")

            # Reset time
            if recorded_episodes < config.num_episodes and not events["stop_recording"]:
                log_say(
                    f"Reset environment ({config.reset_time_s}s)", config.play_sounds
                )
                reset_start = time.perf_counter()
                while time.perf_counter() - reset_start < config.reset_time_s:
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break
                    time.sleep(0.1)

        # Finalize dataset
        dataset.finalize()

        log_say("Recording complete", config.play_sounds, blocking=True)
        logger.info(f"Recorded {recorded_episodes} episodes")
        logger.info(f"Dataset saved to: {dataset_path}")

    finally:
        # Cleanup
        if arm.is_connected:
            arm.disconnect()
        for cam in cameras.values():
            cam.disconnect()
        if not is_headless() and listener:
            listener.stop()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> CollectAlignedConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SO-101 Aligned Data Collection (with offline frame alignment)",
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
        default=DEFAULT_DATASET_FPS,
        help="Output dataset FPS",
    )
    parser.add_argument(
        "--proprioception-fps",
        type=int,
        default=DEFAULT_PROPRIOCEPTION_FPS,
        help="Proprioception sampling rate (Hz)",
    )
    parser.add_argument(
        "--camera-latency-ms",
        type=float,
        default=DEFAULT_CAMERA_LATENCY_MS,
        help="Known camera latency to subtract (ms)",
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

    return CollectAlignedConfig(
        task=Task(args.task),
        setup=Setup(args.setup),
        num_episodes=args.num_episodes,
        dataset_fps=args.dataset_fps,
        proprioception_fps=args.proprioception_fps,
        camera_latency_ms=args.camera_latency_ms,
        episode_time_s=args.episode_time,
        reset_time_s=args.reset_time,
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
    collect_aligned(config)


if __name__ == "__main__":
    main()
