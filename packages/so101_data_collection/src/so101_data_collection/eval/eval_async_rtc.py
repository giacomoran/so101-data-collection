#!/usr/bin/env python
"""Asynchronous ACT Relative RTC policy evaluation with Real-Time Chunking (RTC).

This script extends eval_async_discard.py to support Real-Time Chunking (RTC),
where the policy is conditioned on an action prefix (the first `delay` steps of
the action chunk being executed during inference). This enables smoother chunk
transitions without discontinuities.

Uses act_relative_rtc_2 policy which:
- Uses relative joint positions (action - proprio_obs) for action representation
- Does NOT use delta observation in the encoder (V2 simplification)
- Handles absolute-to-relative prefix conversion internally in predict_action_chunk

Training/inference consistency:
- At training time, proprio_obs[t] = action[t] (hand-guided setup, same arm). The model
  uses obs_state_t (actual observation) as reference for computing relative actions.
- At inference time, we use the actual robot observation as proprio_obs, matching how
  obs_state_t is used during training. The obs == action property is naturally maintained
  when the robot tracks commanded positions well (enabled by execution latency compensation).
- Execution latency is compensated by indexing chunks in execution time and sending
  actions ahead by the measured latency

Key differences from eval_async_discard.py:
- Action prefix is passed to the policy during inference
- The prefix consists of the next `delay` actions that will execute during inference
- The policy converts absolute prefix to relative internally (V2)
- The policy returns a normalized relative action chunk

Architecture:
- Actor thread: runs at fps, switches to new chunks when available, passes action prefix
- Inference thread: triggered by actor, runs inference with absolute prefix
- Shared state: current observation, pending chunk, inference status, RTC prefix data
- No action queue merging needed - clean chunk switches

Usage:
    python -m so101_data_collection.eval.eval_async_rtc \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --robot.id=arm_follower_0 \
        --policy.path=outputs/model_with_rtc/pretrained_model \
        --dataset_repo_id=giacomoran/cube_hand_guided \
        --fps=30 \
        --timesteps_execution_latency=3 \
        --display_data=true
"""

import logging
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: F401
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot_policy_act_relative_rtc_2 import ACTRelativeRTCConfig  # noqa: F401

from so101_data_collection.eval.rerun_utils import (
    log_rerun_data,
)
from so101_data_collection.eval.trackers import LatencyTracker

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalAsyncRTCConfig:
    """Configuration for asynchronous policy evaluation with Real-Time Chunking."""

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

    # Execution latency in control frames (e.g., 3 frames at 30fps ~ 100ms)
    timesteps_execution_latency: int = 0

    # Remaining actions threshold: trigger inference when remaining actions drops below this
    # Should be higher than inference latency + execution latency (in frames)
    # to ensure new chunk is ready before we need to send actions for future exec times.
    # e.g., if inference takes ~200ms (6 frames at 30fps), execution latency is 3 frames,
    # and threshold is 10, we have 1 frame buffer.
    threshold_remaining_actions: int = 8

    # Display and feedback
    display_data: bool = False
    fps_display_data: int = 30
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

        # Validate fps_display_data is multiple of fps
        if self.fps_display_data % self.fps != 0:
            raise ValueError(f"fps_display_data ({self.fps_display_data}) must be a multiple of fps ({self.fps})")
        if self.timesteps_execution_latency < 0:
            raise ValueError("timesteps_execution_latency must be >= 0")

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

    Timesteps are in execution time (aligned with observations). The actor sends
    actions for (timestep + latency_frames) each control frame to compensate for
    robot latency, so actions execute at the intended timestep.

    Actions are stored as ABSOLUTE joint positions (ready to send to robot).
    The proprio_obs field records the reference state used to convert relative
    actions to absolute:
        absolute_action = relative_action + proprio_obs

    For consistency with training (where proprio_obs[t] = action[t] in hand-guided
    setup), proprio_obs is the commanded action from the previous chunk at the
    timestep when inference was requested. For the first chunk (no previous chunk),
    it falls back to the real robot position.

    The policy produces a chunk of actions from an observation and an action
    prefix. `proprio_obs` represents the observation at timestep `timestep_obs`,
    `actions` represents the output chunk of actions, and `delay` represents
    the length (in timesteps) of the action prefix.
    In the ACTRelativeRTC policy:
    - Both at training and inference time, the encoder input is only the
      observation images (not proprioception). Proprioception (proprio_obs) is
      only used for computing relative actions.
    - The predicted action chunk corresponds to timesteps:
      `[timestep_obs + 1 + delay, timestep_obs + 1 + delay + chunk_size)`.
      The `+ 1` is there because in the hand-guided setup action[t] = proprio_obs[t]
      (same arm), so relative_action[timestep_obs] = 0 always and is skipped.

    Attributes:
        actions: Action tensor [n_actions, action_dim] as ABSOLUTE joint positions
        proprio_obs: Actual robot observation [1, state_dim] at inference time
        timestep_obs: Reference execution timestep for action indexing
        delay: Number of prefix steps used; actions[0] executes at timestep_obs + delay + 1
        idx_chunk: Inference counter that generated this chunk (for logging)
    """

    actions: torch.Tensor
    proprio_obs: torch.Tensor
    timestep_obs: int
    delay: int
    idx_chunk: int

    def action_at(self, timestep: int) -> torch.Tensor | None:
        """Get action for the given timestep.

        Args:
            timestep: Timestep to get action for

        Returns:
            Action tensor if timestep is within chunk range, None otherwise
        """
        timestep_first_action = self.timestep_obs + self.delay + 1
        idx = timestep - timestep_first_action
        if 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None

    def count_remaining_actions_from(self, timestep: int) -> int:
        """Get number of actions remaining after the given timestep.

        Args:
            timestep: Timestep to check from

        Returns:
            Number of actions that will execute at timesteps > `timestep`
        """
        timestep_first_action = self.timestep_obs + self.delay + 1
        idx = timestep + 1 - timestep_first_action
        return max(0, len(self.actions) - idx)

    def action_prefix_at(self, timestep: int, length_max: int) -> tuple[torch.Tensor | None, int]:
        """Get action prefix for RTC inference starting after the given timestep.

        At training time, the model is conditioned on action prefix starting from
        timestep_obs + 1 (action[timestep_obs] is skipped because relative_action
        at timestep_obs = 0 in the hand-guided setup). At inference, we match this
        convention: the prefix starts from timestep + 1, not timestep.
        The timestep is in execution-time coordinates (aligned with observations).

        Args:
            timestep: Current timestep (new observation time). Prefix will contain
                actions at timesteps [timestep + 1, timestep + delay].
            length_max: Maximum prefix length (e.g., rtc_max_delay)

        Returns:
            Tuple of (action_prefix, delay) where:
            - action_prefix: Tensor [1, delay, action_dim] or None if delay=0
            - delay: Actual prefix length used (actions from timestep+1 to timestep+delay)
        """
        count_remaining = self.count_remaining_actions_from(timestep)
        delay = min(length_max, count_remaining)

        if delay > 0:
            timestep_first_action = self.timestep_obs + self.delay + 1
            idx_start = timestep + 1 - timestep_first_action
            action_prefix = self.actions[idx_start : idx_start + delay]
            action_prefix = action_prefix.unsqueeze(0)
            return action_prefix, delay
        else:
            return None, 0


# ============================================================================
# Shared State
# ============================================================================


@dataclass
class State:
    """Centralized state for robot operations and thread communication."""

    # Lock for thread-safe access
    # Note: Using Optional[Lock] instead of Lock | None to avoid dataclass issues
    lock: Optional[Lock] = None

    # Current execution timestep (updated by actor each control frame)
    timestep: int = 0

    # Current observation (updated by actor each control frame)
    dict_obs: dict | None = None

    # Active chunk for current execution time (used by inference for prefix)
    action_chunk_active: ActionChunk | None = None

    # Pending chunk from inference thread (future execution times)
    action_chunk_pending: ActionChunk | None = None

    # Inference request state
    #
    # Why we store timestep/obs at request time instead of using current state:
    # Thread scheduling can delay when the inference thread wakes up after being
    # signaled. If inference reads current state.timestep when it wakes (potentially
    # several frames later), the computed prefix and new chunk timing would be based
    # on a later timestep than when inference was actually needed. This could cause
    # the new chunk's first action to fall outside the old chunk's range, creating
    # a gap with no action. By capturing timestep and observation at request time,
    # we ensure deterministic timing: new chunk's first action is always at
    # timestep_inference_requested + delay + 1, which is guaranteed to be within
    # the old chunk's range (since we triggered when old chunk had threshold remaining).
    #
    # These fields are set together when inference is requested, and cleared together
    # when inference completes. When timestep_inference_requested is not None,
    # inference is considered "running" (replaces the old is_inference_running flag).
    timestep_inference_requested: int | None = None
    dict_obs_inference_requested: dict | None = None

    # Event to wake up inference thread
    event_inference_requested: Event | None = None

    # Shutdown control
    event_shutdown: Event | None = None

    def __post_init__(self):
        self.lock = Lock()
        self.event_inference_requested = Event()
        self.count_total_actions = [0]
        self.event_shutdown = Event()


# ============================================================================
# Actor Thread
# ============================================================================


def thread_actor_fn(
    robot,
    state: State,
    ds_features,
    cfg: EvalAsyncRTCConfig,
) -> None:
    """Actor thread: executes actions from current chunk at target fps.

    Execution latency compensation: action chunks are indexed by execution time
    (aligned with observations). At control timestep t, we send action for
    timestep_exec = t + latency_frames so it executes at the intended time.

    This thread:
    1. Get observation from robot
    2. Acquire lock
    3. Update shared state with observation
    4. Check for pending chunk
    5. Pick action
    6. Trigger inference if needed
    7. Release lock
    8. Execute action if available

    If display_data is enabled:
    - Control pipeline runs at fps
    - Loop runs at fps_display_data (faster)
    - On frames between control frames: just read obs and log to rerun with last sent action
    """
    # Determine effective FPS: fps_display_data if display_data enabled, else fps
    fps_target = cfg.fps_display_data if cfg.display_data else cfg.fps
    # Duration in seconds (precise_sleep expects seconds)
    duration_s_frame_target = 1.0 / fps_target
    count_executed_actions = 0
    action_chunk_active: ActionChunk | None = None

    # Execution latency compensation (in control frames)
    timesteps_execution_latency = cfg.timesteps_execution_latency

    # Compute frame skip ratio: how many display frames per control frame
    num_frames_per_control_frame = cfg.fps_display_data // cfg.fps

    # Track last sent action for display at higher fps
    robot_action_last_executed = None

    try:
        timestep = 0  # Execution-time index (aligned with observations)
        idx_frame = 0
        while not state.event_shutdown.is_set():
            ts_start_frame = time.perf_counter()

            # Determine if we should run the control pipeline on this frame
            is_control_frame = (idx_frame % num_frames_per_control_frame) == 0

            # 1. Get observation from robot (logged to rerun even outside control frames)
            dict_obs = robot.get_observation()

            if is_control_frame:
                # Control pipeline: update state, check chunks, pick action, trigger inference

                timestep_exec = timestep + timesteps_execution_latency

                # 2. Acquire lock
                with state.lock:
                    # 3. Update state with observation
                    state.timestep = timestep
                    state.dict_obs = dict_obs.copy()

                    # 4. Check for pending chunk (promote when it covers current execution timestep)
                    if state.action_chunk_pending:
                        action_chunk_pending = state.action_chunk_pending
                        if action_chunk_pending.action_at(timestep) is not None:
                            action_chunk_active = action_chunk_pending
                            state.action_chunk_active = action_chunk_active  # Set active chunk for inference thread
                            state.action_chunk_pending = None
                            if cfg.debug_timing:
                                logging.info(
                                    f"[ACTOR] Switched to chunk #{action_chunk_active.idx_chunk} "
                                    f"(timestep_obs={action_chunk_active.timestep_obs}, delay={action_chunk_active.delay}, "
                                    f"timestep_action_start={1 + action_chunk_active.delay + action_chunk_active.timestep_obs})"
                                )

                    # 5. Pick action for execution time (timestep_exec)
                    # action_chunk_active covers current execution time (timestep);
                    # action_chunk_for_exec provides the future action we must command now.
                    action = None
                    action_chunk_for_exec = None
                    # Prefer pending chunk once its scheduled start is reached (RTC switch point),
                    # even if the active chunk still overlaps that time.
                    if state.action_chunk_pending is not None:
                        action = state.action_chunk_pending.action_at(timestep_exec)
                        if action is not None:
                            action_chunk_for_exec = state.action_chunk_pending

                    if action is None and action_chunk_active is not None:
                        action = action_chunk_active.action_at(timestep_exec)
                        if action is not None:
                            action_chunk_for_exec = action_chunk_active

                    # 6. Trigger inference (only if no pending chunk waiting to be switched to)
                    # Capture timestep and observation now so inference uses consistent timing
                    # (see State docstring for why this matters)
                    count_remaining_actions = None
                    if action_chunk_for_exec is not None:
                        count_remaining_actions = action_chunk_for_exec.count_remaining_actions_from(timestep_exec)

                    can_request_inference = (
                        state.timestep_inference_requested is None and state.action_chunk_pending is None
                    )
                    has_exec_action = action_chunk_for_exec is not None
                    below_threshold = has_exec_action and count_remaining_actions <= cfg.threshold_remaining_actions
                    should_request_inference = can_request_inference and (not has_exec_action or below_threshold)

                    if should_request_inference:
                        state.timestep_inference_requested = timestep
                        state.dict_obs_inference_requested = dict_obs.copy()
                        state.event_inference_requested.set()

                    idx_chunk_active = action_chunk_active.idx_chunk if action_chunk_active else -1
                    idx_chunk_exec = action_chunk_for_exec.idx_chunk if action_chunk_for_exec else -1
                    idx_chunk_pending = state.action_chunk_pending.idx_chunk if state.action_chunk_pending else -1

                # 7. Release lock

                # 8. Execute action
                if action is not None:
                    action = action.cpu()

                    robot_action = make_robot_action(action.unsqueeze(0), ds_features)

                    robot.send_action(robot_action)
                    count_executed_actions += 1

                    # Store last sent action for display
                    robot_action_last_executed = robot_action

                # Get chunk index for logging and rerun
                idx_chunk = idx_chunk_exec
                count_remaining_actions_log = count_remaining_actions if count_remaining_actions is not None else -1

                if cfg.debug_timing:
                    logging.info(
                        f"[ACTOR] timestep={timestep} | timestep_exec={timestep_exec} | "
                        f"chunk_active={idx_chunk_active} | chunk_exec={idx_chunk_exec} | "
                        f"chunk_pending={idx_chunk_pending} | remaining_after_exec={count_remaining_actions_log} | "
                        f"count_executed_actions={count_executed_actions}"
                    )

                # Increment timestep only on control frames
                timestep += 1
            else:
                # Non-control frame: just get obs and log to rerun
                pass

            # Log to rerun at fps_display_data with last sent action
            if cfg.display_data:
                log_rerun_data(
                    timestep=timestep,
                    idx_chunk=idx_chunk if is_control_frame else None,
                    observation=dict_obs,
                    action=robot_action_last_executed if is_control_frame else None,
                )

            idx_frame += 1

            # Duration in seconds (precise_sleep expects seconds)
            duration_s_frame = time.perf_counter() - ts_start_frame
            precise_sleep(max(0, (duration_s_frame_target - duration_s_frame) - 0.001))

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        state.event_shutdown.set()
        sys.exit(1)

    logging.info(f"[ACTOR] Thread shutting down. Total actions executed: {count_executed_actions}")


# ============================================================================
# Inference Thread
# ============================================================================


def thread_inference_fn(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    state: State,
    device: torch.device,
    cfg: EvalAsyncRTCConfig,
    n_action_steps: int,
    motor_names: list[str],
    camera_names: list[str],
    tracker_latency: LatencyTracker,
    robot_type: str,
) -> None:
    """Inference thread: waits for signal, runs inference with RTC, creates new action chunk.

    Timesteps are in execution-time coordinates (aligned with observations). The
    prefix describes actions that will execute before the new chunk starts.

    This thread:
    1. Waits for event_inference_requested event from actor
    2. Gets latest observation and active chunk from shared state
    3. Computes prefix from active chunk (up to rtc_max_delay)
    4. Runs policy inference with delay and absolute action_prefix
    5. Creates new ActionChunk and stores in shared_state
    6. Repeats until shutdown
    """
    use_amp = policy.config.use_amp
    idx_chunk = 0  # Counter for number of inferences performed

    # Get RTC max delay from policy config (0 if not supported)
    rtc_max_delay = getattr(policy.config, "rtc_max_delay", 0)
    timesteps_execution_latency = cfg.timesteps_execution_latency
    warned_latency_prefix = False

    try:
        while not state.event_shutdown.is_set():
            # Wait for actor to request inference
            if not state.event_inference_requested.wait(timeout=1.0):
                continue
            state.event_inference_requested.clear()

            # Get observation, timestep, and active chunk from when inference was requested
            # (see State docstring for why we use request-time values, not current values)
            with state.lock:
                timestep = state.timestep_inference_requested
                dict_obs = state.dict_obs_inference_requested
                action_chunk_active = state.action_chunk_active

                # Skip if no valid request (shouldn't happen, but be safe)
                if timestep is None or dict_obs is None:
                    state.timestep_inference_requested = None
                    state.dict_obs_inference_requested = None
                    continue

            # Always use actual robot observation as proprio_obs (matches training).
            # During training, obs_state_t is the actual robot observation, and relative
            # actions are computed as (action - obs_state_t). The fact that obs == action
            # in hand-guided data is a property of the data, not a model requirement.
            array_proprio_obs = [dict_obs[motor_name] for motor_name in motor_names]
            proprio_obs = torch.tensor(np.array(array_proprio_obs, dtype=np.float32), device=device).unsqueeze(0)

            # Compute prefix from active chunk (if any)
            if action_chunk_active is not None:
                action_prefix_absolute, delay = action_chunk_active.action_prefix_at(timestep, rtc_max_delay)
                if (
                    timesteps_execution_latency > 0
                    and delay < timesteps_execution_latency
                    and not warned_latency_prefix
                ):
                    logging.warning(
                        "RTC delay (%d) is shorter than execution latency (%d frames). "
                        "Some in-flight actions may not be covered by the prefix. "
                        "Consider increasing threshold_remaining_actions or rtc_max_delay.",
                        delay,
                        timesteps_execution_latency,
                    )
                    warned_latency_prefix = True
            else:
                action_prefix_absolute = None
                delay = 0

            # Check shutdown after potentially blocking
            if state.event_shutdown.is_set():
                break

            # Build observation frame (images only - state comes from actor's proprio_obs)
            observation_frame = {}
            for cam_name in camera_names:
                observation_frame[f"observation.images.{cam_name}"] = dict_obs[cam_name]

            # Run inference
            ts_start_inference = time.perf_counter()

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
            ):
                # Prepare observation and apply preprocessor (for images)
                observation = prepare_observation_for_inference(observation_frame, device, None, robot_type)
                observation = preprocessor(observation)

                # Create inference batch with observation state from actor
                # predict_action_chunk uses proprio_obs for prefix conversion (raw, not normalized)
                inference_batch = dict(observation)
                inference_batch["observation.state"] = proprio_obs

                # Get normalized relative action chunk from policy with RTC params
                # predict_action_chunk expects absolute action_prefix (converts internally)
                relative_actions_normalized = policy.predict_action_chunk(
                    inference_batch,
                    delay=delay,
                    action_prefix=action_prefix_absolute,
                )
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
                # proprio_obs: [batch, state_dim] -> unsqueeze to [batch, 1, state_dim]
                absolute_actions = relative_actions + proprio_obs.unsqueeze(1)

            duration_ms_inference = (time.perf_counter() - ts_start_inference) * 1000

            # Check shutdown after inference
            if state.event_shutdown.is_set():
                break

            # Increment inference idx
            idx_chunk += 1

            # Record latency
            tracker_latency.record(duration_ms_inference, log_to_rerun=cfg.display_data)

            # Create new action chunk and store in shared state
            with state.lock:
                state.action_chunk_pending = ActionChunk(
                    actions=absolute_actions.squeeze(0),  # [n_actions, action_dim]
                    proprio_obs=proprio_obs,  # [1, state_dim]
                    timestep_obs=timestep,  # Reference timestep for action indexing
                    delay=delay,
                    idx_chunk=idx_chunk,
                )
                # Clear inference request (signals inference is complete)
                state.timestep_inference_requested = None
                state.dict_obs_inference_requested = None

            # Log inference completion (only in debug mode)
            if cfg.debug_timing:
                logging.info(
                    f"[INFERENCE] chunk={idx_chunk} | "
                    f"duration_ms_inference={duration_ms_inference:.1f}ms | "
                    f"timestep={timestep} | "
                    f"delay={delay} | "
                    f"timestep_action_start={timestep + delay + 1}"
                )

    except Exception as e:
        logging.error(f"[INFERENCE] Fatal exception: {e}")
        traceback.print_exc()
        state.event_shutdown.set()
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================


@parser.wrap()
def main(cfg: EvalAsyncRTCConfig) -> None:
    """Main entry point for asynchronous policy evaluation with RTC."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Initialize rerun if displaying data
    if cfg.display_data:
        init_rerun(session_name="eval_async_rtc")

    # Setup device
    name_device = cfg.policy.device if cfg.policy.device else "auto"
    device = get_safe_torch_device(name_device)
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

    # Log RTC support
    rtc_max_delay = getattr(policy.config, "rtc_max_delay", 0)
    if rtc_max_delay == 0:
        logging.warning("Policy rtc_max_delay=0, running without RTC prefix conditioning.")
    else:
        logging.info(f"Policy RTC max delay: {rtc_max_delay}")
    timesteps_execution_latency = cfg.timesteps_execution_latency
    if timesteps_execution_latency > 0 and rtc_max_delay < timesteps_execution_latency:
        logging.warning(
            "Execution latency is %d frames but policy rtc_max_delay is %d. "
            "Prefix may not cover all in-flight actions.",
            timesteps_execution_latency,
            rtc_max_delay,
        )

    # Override device processor to use the detected device (cuda/cpu/mps)
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create tracker
    latency_tracker = LatencyTracker()

    # Setup signal handler for graceful shutdown
    ProcessSignalHandler(use_threads=True, display_pid=False)

    # Setup keyboard listener for early termination
    keyboard_listener = None
    events_keyboard = {}
    if not is_headless():
        keyboard_listener, events_keyboard = init_keyboard_listener()
        logging.info("Press ESC to terminate episode early")

    # Track threads and state for cleanup
    thread_inference = None
    thread_actor = None
    ts_start_episode = None
    state = None

    try:
        # Connect robot
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        # Run episode
        log_say("Starting evaluation", cfg.play_sounds)

        logging.info(f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps} fps)")
        logging.info(f"Execution latency: {cfg.timesteps_execution_latency} frames")
        logging.info(f"Remaining actions threshold: {cfg.threshold_remaining_actions}")

        # Reset policy and processors
        policy.reset()
        preprocessor.reset()
        latency_tracker.reset()

        # Create shared state
        state = State()

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
        thread_inference = Thread(
            target=thread_inference_fn,
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
                robot_type,
            ),
            daemon=True,
            name="Inference",
        )
        thread_inference.start()

        # Start actor thread
        thread_actor = Thread(
            target=thread_actor_fn,
            args=(
                robot,
                state,
                ds_features,
                cfg,
            ),
            daemon=True,
            name="Actor",
        )
        thread_actor.start()

        # Main thread monitors for termination
        ts_start_episode = time.perf_counter()

        while not state.event_shutdown.is_set():
            duration_s_episode = time.perf_counter() - ts_start_episode

            # Check termination conditions
            if duration_s_episode >= cfg.episode_time_s:
                break

            # Check for keyboard-initiated early exit
            if events_keyboard.get("exit_early", False):
                logging.info("Terminating episode early (ESC pressed)")
                events_keyboard["exit_early"] = False
                break

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Signal threads to shutdown
        if state is not None:
            state.event_shutdown.set()
            state.event_inference_requested.set()

        # Stop keyboard listener FIRST (before joining threads)
        # This prevents listener from catching keys after we exit
        if keyboard_listener is not None and not is_headless():
            logging.info("Stopping keyboard listener...")
            keyboard_listener.stop()
            # Remove reference to prevent any lingering callbacks
            keyboard_listener = None
            events_keyboard.clear()

        # Wait for threads to finish with timeout to prevent hanging
        logging.info("Waiting for inference thread to finish...")
        if thread_inference is not None:
            if thread_inference.is_alive():
                thread_inference.join(timeout=2.0)
                if thread_inference.is_alive():
                    logging.warning("Inference thread did not finish within timeout")
                else:
                    logging.info("Inference thread finished")
            else:
                logging.info("Inference thread already finished")

        logging.info("Waiting for actor thread to finish...")
        if thread_actor is not None:
            if thread_actor.is_alive():
                thread_actor.join(timeout=2.0)
                if thread_actor.is_alive():
                    logging.warning("Actor thread did not finish within timeout")
                else:
                    logging.info("Actor thread finished")
            else:
                logging.info("Actor thread already finished")

        # Log final stats
        if ts_start_episode is not None and state is not None:
            duration_s_episode = time.perf_counter() - ts_start_episode
            logging.info(f"Episode completed in {duration_s_episode:.1f}s")

            # Log data to rerun
            if cfg.display_data:
                latency_tracker.log_summary_to_rerun()

            # Print stats
            latency_stats = latency_tracker.get_stats()
            if latency_stats:
                logging.info(
                    f"Inference latency: mean={latency_stats['mean']:.1f}ms, "
                    f"std={latency_stats['std']:.1f}ms, p95={latency_stats['p95']:.1f}ms"
                )

        # Run homing sequence to safely park the arm before disconnecting
        if robot and robot.is_connected:
            logging.info("Running homing sequence...")
            try:
                # Lazy import to avoid potential import-time side effects
                from so101_data_collection.zxtra.homing import run_homing_sequence

                run_homing_sequence(robot, enable_rerun_logging=False)
            except Exception as e:
                logging.error(f"Homing failed: {e}")

            logging.info("Disconnecting robot...")
            robot.disconnect()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
