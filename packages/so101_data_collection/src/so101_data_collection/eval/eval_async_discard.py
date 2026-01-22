#!/usr/bin/env python
"""Asynchronous ACT Relative RTC policy evaluation with clean chunk switching.

This script runs policy inference asynchronously in a separate thread while the
robot continues executing actions from the current chunk. When a new chunk becomes
available, it switches to it entirely - no merging of leftover actions is needed.

The key insight is that by triggering inference early enough (when remaining actions
in the current chunk drop below a threshold > inference latency), we can guarantee
the new chunk is ready before the old one runs out. This eliminates the complexity of
tracking chunk indices and merging leftover actions.

Architecture:
- Actor thread: runs at fps, switches to new chunks when available
- Inference thread: triggered by actor when action count drops below threshold
- Shared state: current observation, pending chunk, inference status
- No action queue merging needed - clean chunk switches

Usage:
    python -m so101_data_collection.eval.eval_async_discard_3 \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --robot.id=arm_follower_0 \
        --policy.path=outputs/cube_hand_guided_act_umi_wrist_7_16k/pretrained_model_migrated \
        --dataset_repo_id=giacomoran/cube_hand_guided \
        --fps=30 \
        --display_data_fps=60 \
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
from pathlib import Path
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Optional

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
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: F401
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot_policy_act_relative_rtc import ACTRelativeRTCConfig  # noqa: F401

from so101_data_collection.eval.rerun_utils import (
    log_rerun_data,
    plot_observation_actions_from_rerun,
)
from so101_data_collection.eval.trackers import LatencyTracker

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalAsyncDiscardConfig:
    """Configuration for asynchronous policy evaluation with plan switching."""

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

    # Remaining actions threshold: trigger inference when remaining actions drops below this
    # Should be higher than inference delay + execution horizon to ensure new chunk is ready
    # e.g., if inference takes ~200ms (6 timesteps at 30fps) and threshold is 8, we have 2 timesteps buffer
    remaining_actions_threshold: int = 8

    # Display and feedback
    display_data: bool = False
    display_data_fps: int = 30
    play_sounds: bool = True

    # Debug options
    debug_timing: bool = False

    def __post_init__(self):
        # Parse policy path from CLI if provided
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy must be provided via --policy.path=...")

        # Validate display_data_fps is multiple of fps
        if self.display_data_fps % self.fps != 0:
            raise ValueError(f"display_data_fps ({self.display_data_fps}) must be a multiple of fps ({self.fps})")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enable the parser to load config from policy using --policy.path=..."""
        return ["policy"]


# ============================================================================
# Action Chunk
# ============================================================================


@dataclass
class ActionChunk:
    """A chunk of actions with associated timing information.

    Attributes:
        actions: Action tensor [n_actions, action_dim]
        obs_timestep: Timestep when actions[0] should execute
        chunk_idx: Inference counter that generated this chunk (for logging)
    """

    actions: torch.Tensor
    obs_timestep: int
    chunk_idx: int

    def get_action_at(self, timestep: int) -> torch.Tensor | None:
        """Get action for the given timestep.

        Args:
            timestep: Timestep to get action for

        Returns:
            Action tensor if timestep is within chunk range, None otherwise
        """
        idx = timestep - self.obs_timestep
        if 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None

    def remaining_from(self, timestep: int) -> int:
        """Get number of actions remaining from the given timestep.

        Args:
            timestep: Timestep to check from

        Returns:
            Number of actions that will execute at or after this timestep
        """
        idx = timestep - self.obs_timestep
        return max(0, len(self.actions) - idx)


# ============================================================================
# Shared State
# ============================================================================


@dataclass
class State:
    """Centralized state for robot operations and thread communication."""

    obs_delta_frames: int = 1

    # Current observation and its timestep (updated by actor)
    current_obs: dict | None = None
    obs_timestep: int = 0

    # Pending chunk from inference thread (switched to by actor)
    pending_chunk: ActionChunk | None = None

    # Inference control (actor requests, inference thread runs)
    inference_running: bool = False

    # Observation history for delta computation (shared between threads)
    obs_history: deque | None = None

    # Shutdown event (shared across main, actor, and inference threads)
    shutdown_event: Event | None = None

    # Total actions counter (updated by actor, read by main)
    total_actions_counter: list[int] | None = None

    # Lock for thread-safe access
    lock: Optional[Lock] = None

    inference_requested: Event | None = None

    def __post_init__(self):
        self.obs_history = deque(maxlen=self.obs_delta_frames + 1)
        self.lock = Lock()
        self.inference_requested = Event()
        self.total_actions_counter = [0]
        self.shutdown_event = Event()


# ============================================================================
# Inference Thread
# ============================================================================


def inference_thread_fn(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    state: State,
    device: torch.device,
    cfg: EvalAsyncDiscardConfig,
    n_action_steps: int,
    motor_names: list[str],
    camera_names: list[str],
    latency_tracker: LatencyTracker,
    obs_state_delta_frames: int,
    robot_type: str,
) -> None:
    """Inference thread: waits for signal, runs inference, creates new action chunk.

    This thread:
    1. Waits for inference_requested event from actor
    2. Gets latest observation from shared state
    3. Runs policy inference with delta computation
    4. Creates new ActionChunk and stores in shared_state
    5. Repeats until shutdown

    The observation history is maintained by the actor thread for delta computation.
    """
    use_amp = policy.config.use_amp
    chunk_idx = 0  # Counter for number of inferences performed

    try:
        while not state.shutdown_event.is_set():
            # Wait for actor to request inference
            if not state.inference_requested.wait(timeout=1.0):
                continue
            state.inference_requested.clear()

            # Get current observation, timestep, and compute delta from history
            with state.lock:
                raw_obs = state.current_obs
                obs_timestep = state.obs_timestep

                # Skip if observation not available
                if raw_obs is None:
                    state.inference_running = False
                    continue

                # Compute delta observation from history (most recent - oldest)
                # Only compute if we have enough history
                if len(state.obs_history) >= 2:
                    delta_obs = state.obs_history[-1] - state.obs_history[0]
                    current_state = state.obs_history[-1]
                else:
                    # Not enough history, skip this inference and reset flag so actor can retry
                    state.inference_running = False
                    continue

            # Check shutdown after potentially blocking
            if state.shutdown_event.is_set():
                break

            # Build observation frame
            observation_frame = {}
            state_values = [raw_obs[motor_name] for motor_name in motor_names]
            observation_frame["observation.state"] = np.array(state_values, dtype=np.float32)
            for cam_name in camera_names:
                observation_frame[f"observation.images.{cam_name}"] = raw_obs[cam_name]

            # Run inference
            inference_start_time = time.perf_counter()

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
            ):
                # Prepare observation and apply preprocessor
                observation = prepare_observation_for_inference(observation_frame, device, None, robot_type)
                observation = preprocessor(observation)

                # Normalize delta observation if stats are available
                if policy.has_relative_stats:
                    delta_obs_normalized = policy.delta_obs_normalizer(delta_obs)
                else:
                    delta_obs_normalized = delta_obs

                # Create inference batch with delta observation
                inference_batch = dict(observation)
                inference_batch["observation.state"] = delta_obs_normalized

                # Get normalized relative action chunk from policy
                relative_actions_normalized = policy.predict_action_chunk(inference_batch)
                # Shape: [batch, chunk_size, action_dim]

                # Slice to n_action_steps
                relative_actions_normalized = relative_actions_normalized[:, :n_action_steps, :]

                # Unnormalize relative actions
                if policy.has_relative_stats:
                    relative_actions = policy.relative_action_normalizer.inverse(relative_actions_normalized)
                else:
                    relative_actions = relative_actions_normalized

                # Convert to absolute actions
                # relative_actions: [batch, n_action_steps, action_dim]
                # current_state: [batch, state_dim] -> unsqueeze to [batch, 1, state_dim]
                absolute_actions = relative_actions + current_state.unsqueeze(1)

            inference_ms = (time.perf_counter() - inference_start_time) * 1000

            # Check shutdown after inference
            if state.shutdown_event.is_set():
                break

            # Increment inference idx
            chunk_idx += 1

            # Record latency
            latency_tracker.record(inference_ms, log_to_rerun=cfg.display_data)

            # Create new action chunk and store in shared state
            # The policy assumes that action[0] is executed at obs_timestep
            with state.lock:
                actions = absolute_actions.squeeze(0)  # [n_actions, action_dim]
                state.pending_chunk = ActionChunk(
                    actions=actions,
                    obs_timestep=obs_timestep,
                    chunk_idx=chunk_idx,
                )
                state.inference_running = False

            # Log inference completion (only in debug mode)
            if cfg.debug_timing:
                logging.info(
                    f"[INFERENCE] #{chunk_idx} | "
                    f"latency={inference_ms:.1f}ms | "
                    f"obs_t={obs_timestep} | "
                    f"start_t={obs_timestep}"
                )

    except Exception as e:
        logging.error(f"[INFERENCE] Fatal exception: {e}")
        traceback.print_exc()
        state.shutdown_event.set()
        sys.exit(1)


# ============================================================================
# Actor Thread
# ============================================================================


def actor_thread_fn(
    robot,
    state: State,
    ds_features,
    cfg: EvalAsyncDiscardConfig,
    device: torch.device,
    motor_names: list[str],
) -> None:
    """Actor thread: executes actions from current chunk at target fps.

    This thread:
    1. Get observation from robot
    2. Acquire lock
    3. Update shared state with observation
    4. Check for pending chunk
    5. Pick action
    6. Trigger inference if needed
    7. Release lock
    8. Execute action if available

    No action queue merging - clean chunk switches when new chunk is available.

    If display_data is enabled:
    - Control pipeline runs at fps
    - Loop runs at display_data_fps (faster)
    - On frames between control frames: just read obs and log to rerun with last sent action
    """
    # Determine effective FPS: display_data_fps if display_data enabled, else fps
    effective_fps = cfg.display_data_fps if cfg.display_data else cfg.fps
    dt_target = 1.0 / effective_fps
    action_count = 0
    current_chunk: ActionChunk | None = None

    # Precompute execution latency in timesteps (based on control fps)
    dt_ms = 1000.0 / cfg.fps
    execution_latency_timesteps = int(math.ceil(cfg.execution_latency_ms / dt_ms))

    # Compute frame skip ratio: how many display frames per control frame
    frames_per_control = cfg.display_data_fps // cfg.fps

    # Track last sent action for display at higher fps
    last_sent_action = None

    try:
        t = 0
        frame_idx = 0
        chunk_idx = 0
        while not state.shutdown_event.is_set():
            action_start_t = time.perf_counter()

            # Determine if we should run the control pipeline on this frame
            control_frame = (frame_idx % frames_per_control) == 0

            # 1. Get observation from robot
            raw_obs = robot.get_observation()

            if control_frame:
                # Control pipeline: update state, check chunks, pick action, trigger inference
                state_values = [raw_obs[motor_name] for motor_name in motor_names]
                obs_state = torch.tensor(np.array(state_values, dtype=np.float32), device=device).unsqueeze(0)

                effective_t = t + execution_latency_timesteps

                # 2. Acquire lock
                with state.lock:
                    # 3. Update state with observation
                    state.current_obs = raw_obs.copy()
                    state.obs_timestep = t
                    state.obs_history.append(obs_state)

                    # 4. Check for pending chunk
                    if state.pending_chunk:
                        pending = state.pending_chunk
                        if pending.get_action_at(effective_t) is not None:
                            current_chunk = pending
                            state.pending_chunk = None
                            if cfg.debug_timing:
                                logging.info(
                                    f"[ACTOR] Switched to chunk #{current_chunk.chunk_idx} "
                                    f"(start_t={current_chunk.obs_timestep})"
                                )

                    # 5. Pick action
                    action_tensor = None
                    if current_chunk is not None:
                        action_tensor = current_chunk.get_action_at(effective_t)

                    # 6. Trigger inference
                    remaining = current_chunk.remaining_from(effective_t) if current_chunk else 0
                    if not state.inference_running and remaining < cfg.remaining_actions_threshold:
                        state.inference_running = True
                        state.inference_requested.set()

                # 7. Release lock

                # 8. Execute action
                if action_tensor is not None:
                    action_tensor = action_tensor.cpu()

                    robot_action = make_robot_action(action_tensor.unsqueeze(0), ds_features)

                    robot.send_action(robot_action)
                    action_count += 1
                    state.total_actions_counter[0] = action_count

                    # Store last sent action for display
                    last_sent_action = robot_action

                # Increment timestep only on control frames
                t += 1

                # Get chunk index for logging
                chunk_idx = current_chunk.chunk_idx if current_chunk else -1

                if cfg.debug_timing:
                    chunk_info = f"chunk=#{current_chunk.chunk_idx}" if current_chunk else "chunk=None"
                    remaining_info = (
                        f"remaining={current_chunk.remaining_from(effective_t)}" if current_chunk else "remaining=0"
                    )
                    logging.info(
                        f"[ACTOR] t={t} | "
                        f"effective_t={effective_t} | "
                        f"{chunk_info} | "
                        f"{remaining_info} | "
                        f"count={action_count}"
                    )
            else:
                # Non-control frame: just get obs and log to rerun
                pass

            # Log to rerun at display_data_fps with last sent action
            if cfg.display_data:
                log_rerun_data(
                    t=t,
                    observation=raw_obs,
                    action=last_sent_action if control_frame else None,
                    chunk_idx=chunk_idx if control_frame else None,
                )

            frame_idx += 1

            action_duration = time.perf_counter() - action_start_t
            precise_sleep(max(0, (dt_target - action_duration) - 0.001))

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        state.shutdown_event.set()
        sys.exit(1)

    logging.info(f"[ACTOR] Thread shutting down. Total actions executed: {action_count}")


# ============================================================================
# Main Entry Point
# ============================================================================


@parser.wrap()
def main(cfg: EvalAsyncDiscardConfig) -> None:
    """Main entry point for asynchronous policy evaluation with chunk switching."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Initialize rerun if displaying data
    recording_path = None
    if cfg.display_data:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_path = Path(f"outputs/eval/eval_{timestamp}.rrd")

        # Set up recording file to save all logged data
        rr.save(str(recording_path))

        # Initialize rerun for streaming
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
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create tracker
    latency_tracker = LatencyTracker()

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # Setup keyboard listener for early termination
    listener = None
    events = {}
    if not is_headless():
        listener, events = init_keyboard_listener()
        logging.info("Press ESC to terminate episode early")

        # Track threads for cleanup
        inference_thread = None
        actor_thread = None
        start_episode_t = None

    try:
        # Connect robot
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        # Run episode
        log_say("Starting evaluation", cfg.play_sounds)

        logging.info(f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps} fps)")
        logging.info(f"Execution latency: {cfg.execution_latency_ms}ms")
        logging.info(f"Remaining actions threshold: {cfg.remaining_actions_threshold}")

        # Reset policy and processors
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        latency_tracker.reset()

        # Get delta frames config
        obs_state_delta_frames = getattr(policy.config, "obs_state_delta_frames", 1)

        # Precompute timing constants
        dt_ms = 1000.0 / cfg.fps
        execution_latency_timesteps = int(math.ceil(cfg.execution_latency_ms / dt_ms))

        # Log computed values
        logging.info(f"Execution latency: {execution_latency_timesteps} timesteps")
        logging.info(f"Delta frames: {obs_state_delta_frames}")

        # Create shared state
        state = State(obs_delta_frames=obs_state_delta_frames)

        # Setup rerun styling
        if cfg.display_data:
            rr.log(
                "metrics/inference_latency",
                rr.SeriesLines(names="Inference (ms)", colors=[255, 100, 100]),
                static=True,
            )

        # Get dataset features for action conversion
        ds_features = ds_meta.features

        # Get motor names from robot (keys like "shoulder_pan.pos")
        motor_names = [k for k in robot.observation_features if robot.observation_features[k] is float]

        # Get camera names from robot (keys like "wrist")
        camera_names = [k for k in robot.observation_features if isinstance(robot.observation_features[k], tuple)]

        # Get robot type for inference thread
        robot_type = robot.robot_type

        # Get policy config values (with optional override)
        policy_n_action_steps = getattr(policy.config, "n_action_steps", 1)
        n_action_steps = cfg.n_action_steps if cfg.n_action_steps is not None else policy_n_action_steps

        # Start inference thread
        inference_thread = Thread(
            target=inference_thread_fn,
            args=(
                policy,
                preprocessor,
                state,
                device,
                cfg,
                n_action_steps,
                motor_names,
                camera_names,
                latency_tracker,
                obs_state_delta_frames,
                robot_type,
            ),
            daemon=True,
            name="Inference",
        )
        inference_thread.start()

        # Start actor thread
        actor_thread = Thread(
            target=actor_thread_fn,
            args=(
                robot,
                state,
                ds_features,
                cfg,
                device,
                motor_names,
            ),
            daemon=True,
            name="Actor",
        )
        actor_thread.start()

        # Main thread monitors for termination
        start_episode_t = time.perf_counter()

        while not state.shutdown_event.is_set():
            elapsed = time.perf_counter() - start_episode_t

            # Check termination conditions
            if elapsed >= cfg.episode_time_s:
                break

            # Check for keyboard-initiated early exit
            if events.get("exit_early", False):
                logging.info("Terminating episode early (ESC pressed)")
                events["exit_early"] = False
                break

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Signal threads to shutdown
        if "state" in locals():
            state.shutdown_event.set()
            state.inference_requested.set()

        # Stop keyboard listener FIRST (before joining threads)
        # This prevents listener from catching keys after we exit
        if listener is not None and not is_headless():
            logging.info("Stopping keyboard listener...")
            listener.stop()
            # Remove reference to prevent any lingering callbacks
            listener = None
            events.clear()

        # Wait for threads to finish with timeout to prevent hanging
        logging.info("Waiting for inference thread to finish...")
        if inference_thread is not None:
            if inference_thread.is_alive():
                inference_thread.join(timeout=2.0)
                if inference_thread.is_alive():
                    logging.warning("Inference thread did not finish within timeout")
                else:
                    logging.info("Inference thread finished")
            else:
                logging.info("Inference thread already finished")

        logging.info("Waiting for actor thread to finish...")
        if actor_thread is not None:
            if actor_thread.is_alive():
                actor_thread.join(timeout=2.0)
                if actor_thread.is_alive():
                    logging.warning("Actor thread did not finish within timeout")
                else:
                    logging.info("Actor thread finished")
            else:
                logging.info("Actor thread already finished")

        # Log final stats
        if start_episode_t is not None:
            total_time = time.perf_counter() - start_episode_t
            total_actions = state.total_actions_counter[0] if "state" in locals() else 0
            actual_fps = total_actions / total_time if total_time > 0 else 0
            logging.info(f"Episode complete: {total_actions} actions in {total_time:.1f}s ({actual_fps:.1f} fps)")

            # Log data to rerun
            if "cfg" in locals() and cfg.display_data and recording_path is not None:
                latency_tracker.log_summary_to_rerun()
                logging.info(f"Rerun recording saved to {recording_path}")
                logging.info("Export recording from ReRun to create plot")

            # Print stats
            if "latency_tracker" in locals():
                latency_stats = latency_tracker.get_stats()

                if latency_stats:
                    logging.info(
                        f"Inference latency: mean={latency_stats['mean']:.1f}ms, "
                        f"std={latency_stats['std']:.1f}ms, p95={latency_stats['p95']:.1f}ms"
                    )

        # Cleanup
        if robot and robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
