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
        --display_data=true \
        --debug_timing=true
"""

import logging
import math
import sys
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
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot_policy_act_relative_rtc import ACTRelativeRTCConfig  # noqa: F401

from so101_data_collection.eval.rerun_utils import log_rerun_data
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

    # Queue threshold: run inference when queue size drops to this value
    # Should be higher than inference delay + execution horizon
    action_queue_size_to_get_new_actions: int = 2

    # Display and feedback
    display_data: bool = False
    play_sounds: bool = True

    # Debug options
    debug_timing: bool = False

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


class ActionQueue:
    """Thread-safe action queue for async inference with action discarding.

    This queue manages action chunks from the inference thread to the actor thread.
    When a new chunk arrives, it replaces the entire queue (discarding leftover
    actions from the previous chunk) and skips actions corresponding to elapsed time.

    Attributes:
        queue: Current action chunk tensor [remaining_actions, action_dim]
        action_idx: Index of next action to execute
        chunk_start_time: Time when the current chunk was inserted
        chunk_idx: Inference counter for the current chunk
    """

    def __init__(self):
        self.queue: torch.Tensor | None = None
        self.action_idx: int = 0
        self.chunk_start_time: float = 0.0
        self.chunk_idx: int = 0
        self.lock = Lock()

    def replace(
        self,
        action_chunk: torch.Tensor,
        n_skip: int,
        chunk_idx: int,
    ) -> int:
        """Replace queue with new action chunk, skipping first n_skip actions.

        Args:
            action_chunk: New action chunk [batch, n_actions, action_dim]
            n_skip: Number of actions to skip at the start
            chunk_idx: Inference counter for this chunk

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
            self.chunk_idx = chunk_idx

            return leftover + n_skip

    def get(self) -> tuple[torch.Tensor, int, int] | None:
        """Get the next action from the queue.

        Returns:
            Tuple of (action_tensor, chunk_idx, action_idx) or None if queue is empty.
            - action_tensor: Action tensor [action_dim]
            - chunk_idx: Which inference generated this action
            - action_idx: Position within that inference's action sequence (0-indexed)
        """
        with self.lock:
            if self.queue is None or self.action_idx >= len(self.queue):
                return None

            action = self.queue[self.action_idx].clone()
            action_idx = self.action_idx
            chunk_idx = self.chunk_idx
            self.action_idx += 1
            return action, chunk_idx, action_idx

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
    action_queue: ActionQueue,
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
    1. Waits until queue size drops below threshold
    2. Gets observation from robot
    3. Runs policy inference (reads from observation queue, doesn't update it)
    4. Computes how many actions to skip
    5. Replaces the action queue with new chunk (discarding stale actions)
    6. Repeats until shutdown

    The actor thread continues executing from the queue while this thread computes.
    The observation queue is only updated by the actor thread.
    """
    use_amp = policy.config.use_amp
    chunk_idx = 0  # Counter for number of inferences performed

    try:
        while not shutdown_event.is_set():
            # Wait until queue size drops below threshold
            if action_queue.qsize() > cfg.action_queue_size_to_get_new_actions:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue

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

            # Run inference (inline)
            inference_start_time = time.perf_counter()

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type)
                if device.type == "cuda" and use_amp
                else nullcontext(),
            ):
                # Step 1 & 2: Convert to tensors and apply preprocessor
                observation = prepare_observation_for_inference(
                    observation_frame, device, None, robot.robot_type
                )
                observation = preprocessor(observation)

                # Step 3: Compute delta from observation queue (read-only, don't update)
                # The queue is updated by the actor thread at every fps tick
                # Observations are already on GPU (moved there by actor thread)
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
                relative_actions_normalized = policy.predict_action_chunk(
                    inference_batch
                )
                # Shape: [batch, chunk_size, action_dim]

                # Slice to n_action_steps
                relative_actions_normalized = relative_actions_normalized[
                    :, :n_action_steps, :
                ]

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

            inference_ms = (time.perf_counter() - inference_start_time) * 1000

            # Check shutdown after inference
            if shutdown_event.is_set():
                break

            # Increment inference idx
            chunk_idx += 1

            # Record latency
            latency_tracker.record(inference_ms, log_to_rerun=cfg.display_data)

            # Compute how many actions to skip
            n_skip = compute_actions_to_skip(
                inference_ms=inference_ms,
                execution_latency_ms=cfg.execution_latency_ms,
                fps=cfg.fps,
            )

            # Replace queue with new chunk (discarding stale actions)
            # Note: Actions stay on GPU/device until actor thread moves them to CPU
            total_discarded = action_queue.replace(absolute_actions, n_skip, chunk_idx)

            # Record discarded count
            discard_tracker.record(total_discarded, log_to_rerun=cfg.display_data)

            # Log inference completion (only in debug mode)
            if cfg.debug_timing:
                logging.info(
                    f"[INFERENCE] #{chunk_idx} | "
                    f"latency={inference_ms:.1f}ms | "
                    f"actions_to_skip={n_skip} | "
                    f"actions_discarded={total_discarded}"
                )

    except Exception as e:
        logging.error(f"[INFERENCE] Fatal exception: {e}")
        traceback.print_exc()
        shutdown_event.set()
        sys.exit(1)


# ============================================================================
# Actor Thread
# ============================================================================


def actor_thread_fn(
    robot: RobotWrapper,
    action_queue: ActionQueue,
    obs_queue: ObservationQueue,
    ds_features,
    cfg: EvalAsyncDiscardConfig,
    device: torch.device,
    motor_names: list[str],
    shutdown_event: Event,
    total_actions_counter: list,  # Use list for mutable reference
) -> None:
    """Actor thread: continuously executes actions from the queue at target fps.

    This thread:
    1. Gets next action from queue (or waits if empty)
    2. Sends action to robot (if available)
    3. Updates observation queue at every fps tick (independent of actions)
    4. Sleeps to maintain target fps
    5. Repeats until shutdown

    Observations are independent from actions - the queue is updated every iteration
    to track the continuous state of the robot.
    """
    dt_target = 1.0 / cfg.fps
    action_count = 0
    timestep = 0  # Track timestep counter for logging

    try:
        while not shutdown_event.is_set():
            action_start_t = time.perf_counter()
            timestamp = time.time()  # Wall clock timestamp

            # Get action from queue (returns tuple with metadata)
            action_result = action_queue.get()

            if action_result is not None:
                action_tensor, chunk_idx, action_idx = action_result

                # Move action to CPU (matching lerobot_async_inference_example.py pattern)
                action_tensor = action_tensor.cpu()

                # Convert action tensor to robot action dict
                # make_robot_action uses ds_features[ACTION]["names"] to build keys
                # If dataset action names include '.pos' suffix, robot_action is ready to use
                robot_action = make_robot_action(
                    action_tensor.unsqueeze(0), ds_features
                )

                # Log to rerun
                if cfg.display_data:
                    vis_obs = robot.get_observation()
                    log_rerun_data(observation=vis_obs, action=robot_action)

                # Send action to robot
                robot.send_action(robot_action)
                action_count += 1
                total_actions_counter[0] = action_count

                # Log action execution (only if debug_timing is enabled)
                if cfg.debug_timing:
                    logging.info(
                        f"[ACTOR] timestep={timestep} | "
                        f"timestamp={timestamp:.3f} | "
                        f"chunk_idx={chunk_idx} | "
                        f"action_idx={action_idx} | "
                        f"action_count={action_count}"
                    )
                    timestep += 1

            # Update observation queue at every fps tick (independent of actions)
            # Ensures the queue advances continuously, matching select_action behavior
            # In lerobot-record, select_action is called every step and updates the queue each time.
            # Here we need to manually update the queue every iteration to maintain the same behavior.
            # Move observations to GPU immediately (inference thread will use them directly)
            raw_obs = robot.get_observation()
            state_values = [raw_obs[motor_name] for motor_name in motor_names]
            obs_state = torch.tensor(
                np.array(state_values, dtype=np.float32), device=device
            ).unsqueeze(0)  # Add batch dimension [1, state_dim]
            obs_queue.update(obs_state)

            # Sleep to maintain target fps
            action_duration = time.perf_counter() - action_start_t
            precise_sleep(max(0, (dt_target - action_duration) - 0.001))

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        shutdown_event.set()
        sys.exit(1)

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
    cfg: EvalAsyncDiscardConfig,
    device: torch.device,
    latency_tracker: LatencyTracker,
    discard_tracker: DiscardTracker,
    ds_meta,
    shutdown_event: Event,
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
    action_queue = ActionQueue()

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

    # Initialize observation queue with current observation
    # Store on GPU (actor thread will update with GPU tensors)
    raw_obs_init = robot.get_observation()
    state_values_init = [raw_obs_init[motor_name] for motor_name in motor_names]
    obs_state_init = torch.tensor(
        np.array(state_values_init, dtype=np.float32), device=device
    ).unsqueeze(0)
    obs_queue.update(obs_state_init)

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
                obs_queue,
                ds_features,
                cfg,
                device,
                motor_names,
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
            if elapsed >= cfg.episode_time_s:
                break

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

    finally:
        # Signal threads to shutdown
        shutdown_event.set()

        # Wait for threads to finish
        if inference_thread is not None and inference_thread.is_alive():
            inference_thread.join()

        if actor_thread is not None and actor_thread.is_alive():
            actor_thread.join()

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
    device_str = cfg.policy.device if cfg.policy.device else "auto"
    device = get_safe_torch_device(device_str)
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

    # Override device processor to use the detected device (cuda/cpu/mps)
    # This ensures compatibility when loading models trained on different hardware
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create trackers
    latency_tracker = LatencyTracker()
    discard_tracker = DiscardTracker()

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    try:
        # Connect robot
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        # Run episode
        log_say("Starting evaluation", cfg.play_sounds)
        run_episode_async_discard(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            cfg=cfg,
            device=device,
            latency_tracker=latency_tracker,
            discard_tracker=discard_tracker,
            ds_meta=ds_meta,
            shutdown_event=shutdown_event,
        )

        log_say("Episode finished", cfg.play_sounds, blocking=True)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Cleanup
        if robot and robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
