#!/usr/bin/env python
"""Asynchronous ACT Relative RTC policy evaluation with action discarding.

This script runs policy inference asynchronously in a separate thread while the
robot continues executing actions from the previous chunk. When a new chunk
becomes available:
1. Discard all remaining actions from the previous chunk
2. Discard actions in the new chunk that correspond to elapsed time (inference
   latency + execution latency)

This combines the benefits of async inference (no pauses during inference) with
UMI-style latency compensation (discarding stale actions).

The inference logic is fully decoupled from the policy's internal queue mechanism.
We call predict_action_chunk() directly and handle:
- Observation delta computation (obs[t] - obs[t-N])
- Delta observation normalization
- Relative action unnormalization
- Relative-to-absolute action conversion

Inspired by LeRobot's RTC implementation in examples/rtc/eval_with_real_robot.py.

Usage:
    python -m so101_data_collection.eval.eval_async_discard \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --robot.id=arm_follower_0 \
        --policy.path=outputs/cube_hand_guided_act_umi_wrist_7_16k/pretrained_model_migrated \
        --dataset_repo_id=giacomoran/cube_hand_guided \
        --fps=30 \
        --episode_time_s=60 \
        --execution_latency_ms=100 \
        --display_data=true
"""

import logging
import math
import signal
import time
import traceback
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pprint import pformat
from threading import Event, Lock, Thread

import numpy as np
import rerun as rr
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot_policy_act_relative_rtc import ACTRelativeRTCConfig  # noqa: F401

from so101_data_collection.eval.trackers import DiscardTracker, LatencyTracker

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalAsyncDiscardConfig:
    """Configuration for asynchronous policy evaluation with action discarding."""

    robot: RobotConfig
    # Dataset metadata (required for policy loading)
    # Use the dataset repo_id that was used for training
    dataset_repo_id: str
    policy: PreTrainedConfig | None = None

    # Control parameters
    fps: int = 30
    episode_time_s: float = 60.0

    # Override n_action_steps from policy config (None = use policy default)
    n_action_steps: int | None = None

    # Execution latency: time from sending action to robot until it starts executing
    # Measured empirically (~100ms with std ~10ms for SO101)
    execution_latency_ms: float = 100.0

    # Display and feedback
    display_data: bool = False
    play_sounds: bool = True

    def __post_init__(self):
        # Parse policy path from CLI if provided
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy must be provided via --policy.path=...")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enable the parser to load config from policy using --policy.path=..."""
        return ["policy"]


# ============================================================================
# Thread-Safe Robot Wrapper
# ============================================================================


class RobotWrapper:
    """Thread-safe wrapper around the robot for concurrent access."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict:
        """Get observation from robot (thread-safe)."""
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: dict) -> None:
        """Send action to robot (thread-safe)."""
        with self.lock:
            self.robot.send_action(action)

    @property
    def observation_features(self):
        return self.robot.observation_features

    @property
    def robot_type(self):
        return self.robot.robot_type


# ============================================================================
# Thread-Safe Action Queue
# ============================================================================


class AsyncActionQueue:
    """Thread-safe action queue for async inference with action discarding.

    This queue manages action chunks from the inference thread to the actor thread.
    When a new chunk arrives, it replaces the entire queue (discarding leftover
    actions from the previous chunk) and skips actions corresponding to elapsed time.

    Attributes:
        queue: Current action chunk tensor [remaining_actions, action_dim]
        action_idx: Index of next action to execute
        chunk_start_time: Time when the current chunk was inserted
    """

    def __init__(self):
        self.queue: torch.Tensor | None = None
        self.action_idx: int = 0
        self.chunk_start_time: float = 0.0
        self.lock = Lock()

    def replace(
        self,
        action_chunk: torch.Tensor,
        n_skip: int,
    ) -> int:
        """Replace queue with new action chunk, skipping first n_skip actions.

        Args:
            action_chunk: New action chunk [batch, n_actions, action_dim]
            n_skip: Number of actions to skip at the start

        Returns:
            Number of actions actually discarded (includes leftover from prev chunk)
        """
        with self.lock:
            # Count leftover actions from previous chunk
            leftover = 0
            if self.queue is not None:
                leftover = len(self.queue) - self.action_idx

            # Remove batch dimension and skip first n_skip actions
            # action_chunk shape: [batch, n_actions, action_dim] -> [n_actions, action_dim]
            actions = action_chunk.squeeze(0)

            # Clamp n_skip to leave at least 1 action
            n_skip = min(n_skip, len(actions) - 1)

            self.queue = actions[n_skip:]
            self.action_idx = 0
            self.chunk_start_time = time.perf_counter()

            return leftover + n_skip

    def get(self) -> torch.Tensor | None:
        """Get the next action from the queue.

        Returns:
            Action tensor [action_dim] or None if queue is empty.
        """
        with self.lock:
            if self.queue is None or self.action_idx >= len(self.queue):
                return None

            action = self.queue[self.action_idx].clone()
            self.action_idx += 1
            return action

    def qsize(self) -> int:
        """Get number of remaining actions in the queue."""
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.action_idx

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.qsize() == 0


# ============================================================================
# Observation Queue Management
# ============================================================================


class ObservationQueue:
    """Manages observation history for delta computation (thread-safe).

    Maintains a queue of observations to compute:
        delta_obs = obs[t] - obs[t - obs_state_delta_frames]
    """

    def __init__(self, obs_state_delta_frames: int):
        """Initialize the observation queue.

        Args:
            obs_state_delta_frames: Number of frames to look back for delta.
        """
        self.obs_state_delta_frames = obs_state_delta_frames
        # Queue size = delta_frames + 1 to store [obs[t-N], ..., obs[t]]
        queue_size = obs_state_delta_frames + 1
        self.queue: deque[torch.Tensor] = deque(maxlen=queue_size)
        self.lock = Lock()

    def update(self, obs_state: torch.Tensor) -> None:
        """Add new observation to queue.

        On first call, fills queue with copies of the observation.
        """
        with self.lock:
            if len(self.queue) < self.queue.maxlen:
                # Initialize by copying first observation until queue is full
                while len(self.queue) < self.queue.maxlen:
                    self.queue.append(obs_state.clone())
            else:
                self.queue.append(obs_state)

    def get_delta(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute delta observation.

        Returns:
            Tuple of (delta_obs, obs_state_t) where:
            - delta_obs = obs[t] - obs[t - obs_state_delta_frames]
            - obs_state_t = current observation (needed for absolute action conversion)
        """
        with self.lock:
            obs_list = list(self.queue)
            obs_state_t_minus_n = obs_list[0]  # oldest in queue
            obs_state_t = obs_list[-1]  # current observation
            delta_obs = obs_state_t - obs_state_t_minus_n
            return delta_obs, obs_state_t

    def reset(self) -> None:
        """Clear the queue for a new episode."""
        with self.lock:
            self.queue.clear()


# ============================================================================
# Inference Helpers
# ============================================================================


def warmup_policy(
    robot: RobotWrapper,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    obs_queue: ObservationQueue,
    device: torch.device,
    n_action_steps: int,
    motor_names: list[str],
    camera_names: list[str],
    use_amp: bool,
) -> None:
    """Warm up the policy with a dummy inference to avoid slow first inference during episode.

    This pre-initializes CUDA kernels, model weights, and other first-run overhead.
    """
    logging.info("Warming up policy...")
    raw_obs = robot.get_observation()

    # Build observation frame
    observation_frame = {}
    state_values = [raw_obs[motor_name] for motor_name in motor_names]
    observation_frame["observation.state"] = np.array(state_values, dtype=np.float32)
    for cam_name in camera_names:
        observation_frame[f"observation.images.{cam_name}"] = raw_obs[cam_name]

    # Run a dummy inference
    start_time = time.perf_counter()
    _, _ = run_inference_chunk(
        observation_frame=observation_frame,
        obs_queue=obs_queue,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        n_action_steps=n_action_steps,
        use_amp=use_amp,
        task=None,
        robot_type=robot.robot_type,
    )
    warmup_time_ms = (time.perf_counter() - start_time) * 1000
    logging.info(f"Policy warmup complete ({warmup_time_ms:.1f}ms)")


def run_inference_chunk(
    observation_frame: dict[str, np.ndarray],
    obs_queue: ObservationQueue,
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline,
    n_action_steps: int,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """Run policy inference and return absolute action chunk with timing.

    This function handles the full inference pipeline:
    1. Prepare observation for inference (convert to tensors, add batch dim)
    2. Apply preprocessor (image normalization, device placement)
    3. Update observation queue and compute delta
    4. Normalize delta observation (if policy has relative stats)
    5. Call policy.predict_action_chunk() to get normalized relative actions
    6. Unnormalize relative actions (if policy has relative stats)
    7. Convert relative to absolute: absolute = relative + obs[t]

    Args:
        observation_frame: Raw observation dict with numpy arrays
        obs_queue: ObservationQueue for delta computation
        policy: The ACTRelativeRTCPolicy
        device: Torch device for inference
        preprocessor: Pipeline for observation preprocessing
        n_action_steps: Number of actions to return from the chunk
        use_amp: Whether to use automatic mixed precision
        task: Optional task identifier
        robot_type: Optional robot type identifier

    Returns:
        Tuple of (absolute_action_chunk, inference_time_ms) where:
        - absolute_action_chunk: [batch, n_action_steps, action_dim] tensor
        - inference_time_ms: Time taken for inference in milliseconds
    """
    start_time = time.perf_counter()

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext(),
    ):
        # Step 1 & 2: Convert to tensors and apply preprocessor
        observation = prepare_observation_for_inference(
            observation_frame, device, task, robot_type
        )
        observation = preprocessor(observation)

        # Step 3: Update observation queue and compute delta
        obs_state = observation["observation.state"]  # [batch, state_dim]
        obs_queue.update(obs_state)
        delta_obs, obs_state_t = obs_queue.get_delta()

        # Step 4: Normalize delta observation if stats are available
        if policy.has_relative_stats:
            delta_obs_normalized = policy.delta_obs_normalizer(delta_obs)
        else:
            delta_obs_normalized = delta_obs

        # Create inference batch with delta observation
        inference_batch = dict(observation)
        inference_batch["observation.state"] = delta_obs_normalized

        # Step 5: Get normalized relative action chunk from policy
        relative_actions_normalized = policy.predict_action_chunk(inference_batch)
        # Shape: [batch, chunk_size, action_dim]

        # Slice to n_action_steps
        relative_actions_normalized = relative_actions_normalized[:, :n_action_steps, :]

        # Step 6: Unnormalize relative actions
        if policy.has_relative_stats:
            relative_actions = policy.relative_action_normalizer.inverse(
                relative_actions_normalized
            )
        else:
            relative_actions = relative_actions_normalized

        # Step 7: Convert to absolute actions
        # relative_actions: [batch, n_action_steps, action_dim]
        # obs_state_t: [batch, state_dim] -> unsqueeze to [batch, 1, state_dim]
        absolute_actions = relative_actions + obs_state_t.unsqueeze(1)

    inference_time_ms = (time.perf_counter() - start_time) * 1000
    return absolute_actions, inference_time_ms


def compute_actions_to_skip(
    inference_ms: float,
    execution_latency_ms: float,
    fps: int,
) -> int:
    """Compute how many actions to skip based on total latency.

    The policy assumes action[0] should execute at time t (when observation was taken).
    Due to latency, actual execution starts at t + inference_time + execution_latency.
    We skip actions corresponding to this elapsed time.

    Args:
        inference_ms: Time taken for inference in milliseconds
        execution_latency_ms: Fixed latency from send_action to motor execution
        fps: Target control frequency

    Returns:
        Number of actions to skip (0 means execute from action[0])
    """
    total_latency_ms = inference_ms + execution_latency_ms
    dt_ms = 1000.0 / fps
    n_skip = int(math.ceil(total_latency_ms / dt_ms))
    return n_skip


# ============================================================================
# Inference Thread
# ============================================================================


def inference_thread_fn(
    robot: RobotWrapper,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    obs_queue: ObservationQueue,
    action_queue: AsyncActionQueue,
    device: torch.device,
    cfg: EvalAsyncDiscardConfig,
    n_action_steps: int,
    motor_names: list[str],
    camera_names: list[str],
    latency_tracker: LatencyTracker,
    discard_tracker: DiscardTracker,
    shutdown_event: Event,
) -> None:
    """Inference thread: continuously runs policy inference and updates action queue.

    This thread:
    1. Gets observation from robot
    2. Runs policy inference
    3. Computes how many actions to skip
    4. Replaces the action queue with new chunk (discarding stale actions)
    5. Repeats continuously until shutdown

    The actor thread continues executing from the queue while this thread computes.
    """
    chunk_idx = 0
    last_log_time = time.perf_counter()

    try:
        while not shutdown_event.is_set():
            # Get observation
            raw_obs = robot.get_observation()

            # Check shutdown after potentially blocking operation
            if shutdown_event.is_set():
                break

            # Build observation frame
            observation_frame = {}
            state_values = [raw_obs[motor_name] for motor_name in motor_names]
            observation_frame["observation.state"] = np.array(
                state_values, dtype=np.float32
            )
            for cam_name in camera_names:
                observation_frame[f"observation.images.{cam_name}"] = raw_obs[cam_name]

            # Run inference
            action_chunk, inference_ms = run_inference_chunk(
                observation_frame=observation_frame,
                obs_queue=obs_queue,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                n_action_steps=n_action_steps,
                use_amp=policy.config.use_amp,
                task=None,
                robot_type=robot.robot_type,
            )

            # Check shutdown after inference
            if shutdown_event.is_set():
                break

            # Record latency
            latency_tracker.record(inference_ms, log_to_rerun=cfg.display_data)

            # Compute how many actions to skip
            n_skip = compute_actions_to_skip(
                inference_ms=inference_ms,
                execution_latency_ms=cfg.execution_latency_ms,
                fps=cfg.fps,
            )

            # Move to CPU before putting in queue
            action_chunk_cpu = action_chunk.cpu()

            # Replace queue with new chunk (discarding stale actions)
            total_discarded = action_queue.replace(action_chunk_cpu, n_skip)

            # Record discarded count
            discard_tracker.record(total_discarded, log_to_rerun=cfg.display_data)

            # Log periodically (every 2 seconds or every 50 chunks, whichever comes first)
            queue_size = action_queue.qsize()
            now = time.perf_counter()
            if chunk_idx == 0 or (now - last_log_time) >= 2.0 or chunk_idx % 50 == 0:
                logging.info(
                    f"[INFERENCE] Chunk {chunk_idx}: inference={inference_ms:.1f}ms, "
                    f"skip={n_skip}, discarded={total_discarded}, queue_size={queue_size}"
                )
                last_log_time = now

            chunk_idx += 1

    except Exception as e:
        logging.error(f"[INFERENCE] Fatal exception: {e}")
        traceback.print_exc()
        shutdown_event.set()

    logging.info(f"[INFERENCE] Thread shutting down. Total chunks: {chunk_idx}")


# ============================================================================
# Actor Thread
# ============================================================================


def actor_thread_fn(
    robot: RobotWrapper,
    action_queue: AsyncActionQueue,
    ds_features,
    cfg: EvalAsyncDiscardConfig,
    shutdown_event: Event,
    total_actions_counter: list,  # Use list for mutable reference
) -> None:
    """Actor thread: continuously executes actions from the queue at target fps.

    This thread:
    1. Gets next action from queue (or waits if empty)
    2. Sends action to robot
    3. Sleeps to maintain target fps
    4. Repeats until shutdown
    """
    dt_target = 1.0 / cfg.fps
    action_count = 0
    empty_queue_count = 0
    last_empty_log_time = 0.0

    try:
        while not shutdown_event.is_set():
            action_start_t = time.perf_counter()

            # Get action from queue
            action_tensor = action_queue.get()

            if action_tensor is not None:
                # Convert action tensor to robot action dict
                # make_robot_action returns keys like "shoulder_pan",
                # but robot expects "shoulder_pan.pos"
                action_dict = make_robot_action(action_tensor.unsqueeze(0), ds_features)
                robot_action = {f"{name}.pos": val for name, val in action_dict.items()}

                # Log to rerun
                if cfg.display_data:
                    vis_obs = robot.get_observation()
                    log_rerun_data(observation=vis_obs, action=robot_action)

                # Send action to robot
                robot.send_action(robot_action)
                action_count += 1
                total_actions_counter[0] = action_count
                empty_queue_count = 0  # Reset counter when we have actions
            else:
                # Queue is empty - track this for debugging
                empty_queue_count += 1
                now = time.perf_counter()
                # Log if queue has been empty for more than 0.5 seconds
                if empty_queue_count == 1 or (now - last_empty_log_time) >= 0.5:
                    queue_size = action_queue.qsize()
                    logging.debug(
                        f"[ACTOR] Queue empty (size={queue_size}, "
                        f"empty_count={empty_queue_count})"
                    )
                    last_empty_log_time = now

            # Sleep to maintain target fps
            # Use shorter sleep intervals to respond faster to shutdown
            action_duration = time.perf_counter() - action_start_t
            remaining_sleep = dt_target - action_duration

            # Sleep in small increments to check shutdown more frequently
            while remaining_sleep > 0 and not shutdown_event.is_set():
                sleep_chunk = min(remaining_sleep, 0.01)  # 10ms max
                precise_sleep(sleep_chunk)
                remaining_sleep -= sleep_chunk

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        shutdown_event.set()

    logging.info(
        f"[ACTOR] Thread shutting down. Total actions executed: {action_count}"
    )


# ============================================================================
# Main Evaluation Loop
# ============================================================================


def run_episode_async_discard(
    robot,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    events: dict,
    cfg: EvalAsyncDiscardConfig,
    device: torch.device,
    latency_tracker: LatencyTracker,
    discard_tracker: DiscardTracker,
    ds_meta,
) -> None:
    """Run a single episode with asynchronous inference and action discarding.

    Architecture:
    - Inference thread: continuously runs policy inference, replaces action queue
    - Actor thread: continuously executes actions from queue at target fps
    - Main thread: monitors for termination conditions

    When a new chunk arrives, the action queue is replaced entirely, discarding:
    1. Leftover actions from the previous chunk
    2. Actions in the new chunk corresponding to elapsed time
    """
    logging.info(f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps} fps)")
    logging.info(f"Execution latency: {cfg.execution_latency_ms}ms")

    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    latency_tracker.reset()
    discard_tracker.reset()

    # Create observation queue for delta computation
    obs_state_delta_frames = getattr(policy.config, "obs_state_delta_frames", 1)
    obs_queue = ObservationQueue(obs_state_delta_frames)

    # Create action queue for thread communication
    action_queue = AsyncActionQueue()

    # Setup rerun styling (static, logged once)
    if cfg.display_data:
        rr.log(
            "metrics/inference_latency",
            rr.SeriesLines(names="Inference (ms)", colors=[255, 100, 100]),
            static=True,
        )
        rr.log(
            "metrics/discarded_actions",
            rr.SeriesLines(names="Discarded", colors=[255, 200, 0]),
            static=True,
        )

    # Get dataset features for action conversion
    ds_features = ds_meta.features

    # Get motor names from robot (keys like "shoulder_pan.pos")
    motor_names = [
        k for k in robot.observation_features if robot.observation_features[k] is float
    ]

    # Get camera names from robot (keys like "wrist")
    camera_names = [
        k
        for k in robot.observation_features
        if isinstance(robot.observation_features[k], tuple)
    ]

    # Get policy config values (with optional override)
    policy_n_action_steps = getattr(policy.config, "n_action_steps", 1)
    n_action_steps = (
        cfg.n_action_steps if cfg.n_action_steps is not None else policy_n_action_steps
    )

    # Wrap robot for thread-safe access
    robot_wrapper = RobotWrapper(robot)

    # Warm up policy before starting episode to avoid slow first inference
    warmup_policy(
        robot=robot_wrapper,
        policy=policy,
        preprocessor=preprocessor,
        obs_queue=obs_queue,
        device=device,
        n_action_steps=n_action_steps,
        motor_names=motor_names,
        camera_names=camera_names,
        use_amp=policy.config.use_amp,
    )

    # Create shutdown event for threads
    shutdown_event = Event()

    # Counter for total actions (mutable reference for actor thread)
    total_actions_counter = [0]

    # Track threads for cleanup
    inference_thread = None
    actor_thread = None

    try:
        # Start inference thread
        inference_thread = Thread(
            target=inference_thread_fn,
            args=(
                robot_wrapper,
                policy,
                preprocessor,
                obs_queue,
                action_queue,
                device,
                cfg,
                n_action_steps,
                motor_names,
                camera_names,
                latency_tracker,
                discard_tracker,
                shutdown_event,
            ),
            daemon=True,
            name="Inference",
        )
        inference_thread.start()

        # Start actor thread
        actor_thread = Thread(
            target=actor_thread_fn,
            args=(
                robot_wrapper,
                action_queue,
                ds_features,
                cfg,
                shutdown_event,
                total_actions_counter,
            ),
            daemon=True,
            name="Actor",
        )
        actor_thread.start()

        # Main thread monitors for termination
        start_episode_t = time.perf_counter()

        while not shutdown_event.is_set():
            elapsed = time.perf_counter() - start_episode_t

            # Check termination conditions
            if events.get("exit_early") or events.get("stop_recording"):
                events["exit_early"] = False
                break

            if elapsed >= cfg.episode_time_s:
                break

            # Sleep briefly to avoid busy waiting
            # Use short interval to respond quickly to events
            time.sleep(0.05)

    finally:
        # Signal threads to shutdown
        shutdown_event.set()

        # Wait for threads to finish with timeout
        if inference_thread is not None and inference_thread.is_alive():
            inference_thread.join(timeout=3.0)
            if inference_thread.is_alive():
                logging.warning("Inference thread did not exit cleanly")

        if actor_thread is not None and actor_thread.is_alive():
            actor_thread.join(timeout=3.0)
            if actor_thread.is_alive():
                logging.warning("Actor thread did not exit cleanly")

    # Log final stats
    total_time = time.perf_counter() - start_episode_t
    total_actions = total_actions_counter[0]
    actual_fps = total_actions / total_time if total_time > 0 else 0
    logging.info(
        f"Episode complete: {total_actions} actions in {total_time:.1f}s "
        f"({actual_fps:.1f} fps)"
    )

    # Log summaries to rerun
    if cfg.display_data:
        latency_tracker.log_summary_to_rerun()
        discard_tracker.log_summary_to_rerun()

    # Print stats
    latency_stats = latency_tracker.get_stats()
    discard_stats = discard_tracker.get_stats()

    if latency_stats:
        logging.info(
            f"Inference latency: mean={latency_stats['mean']:.1f}ms, "
            f"std={latency_stats['std']:.1f}ms, p95={latency_stats['p95']:.1f}ms"
        )

    if discard_stats:
        logging.info(
            f"Discarded actions: total={discard_stats['total_discarded']}, "
            f"mean={discard_stats['mean']:.1f}/chunk"
        )


# ============================================================================
# Main Entry Point
# ============================================================================


@parser.wrap()
def main(cfg: EvalAsyncDiscardConfig) -> None:
    """Main entry point for asynchronous policy evaluation with discarding."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Initialize rerun if displaying data
    if cfg.display_data:
        init_rerun(session_name="eval_async_discard")

    # Setup device
    device = get_safe_torch_device(cfg.policy.device)
    logging.info(f"Using device: {device}")

    # Create robot
    logging.info("Creating robot...")
    robot = make_robot_from_config(cfg.robot)

    # Load dataset metadata (required for policy loading)
    logging.info(f"Loading dataset metadata from {cfg.dataset_repo_id}...")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)

    # Create policy and processors
    logging.info(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy = make_policy(cfg.policy, ds_meta=ds_meta, env_cfg=None)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
    )

    # Initialize keyboard listener
    listener, events = init_keyboard_listener()

    # Create trackers
    latency_tracker = LatencyTracker()
    discard_tracker = DiscardTracker()

    # Setup signal handler for graceful shutdown on Ctrl+C
    shutdown_requested = [False]
    original_sigint = signal.getsignal(signal.SIGINT)

    def sigint_handler(signum, frame):
        if shutdown_requested[0]:
            # Second Ctrl+C - force exit
            logging.warning("Force exit requested")
            if original_sigint and callable(original_sigint):
                original_sigint(signum, frame)
            else:
                raise KeyboardInterrupt
        else:
            shutdown_requested[0] = True
            events["exit_early"] = True
            logging.info("Shutdown requested (press Ctrl+C again to force)")

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        # Connect robot
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        # Run episode
        log_say(
            "Starting evaluation",
            cfg.play_sounds,
        )
        run_episode_async_discard(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            events=events,
            cfg=cfg,
            device=device,
            latency_tracker=latency_tracker,
            discard_tracker=discard_tracker,
            ds_meta=ds_meta,
        )

        log_say("Episode finished", cfg.play_sounds, blocking=True)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)

        # Cleanup
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()

        if not is_headless() and listener:
            listener.stop()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
