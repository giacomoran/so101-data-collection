#!/usr/bin/env python
"""Asynchronous ACTSmooth policy evaluation.

ACTSmooth uses action prefix conditioning: instead of waiting for a chunk to complete,
inference is triggered while the current chunk is still executing. The policy is conditioned
on an action prefix (past + committed future actions) and predicts a continuation chunk.

Action Prefix Structure (see modeling_act_smooth.py)
----------------------------------------------------
The prefix consists of two parts:

    Prefix = [past completed actions] + [committed pending actions]
             |---- k actions ---------|  |---- d actions (d >= 1) ---|

- Past (history): Actions already executed. Provides context for continuity.
- Future (committed): Actions that WILL execute during inference latency. The policy predicts
  a continuation starting AFTER these committed actions.

The length d of the future prefix determines when the first predicted action executes
(timestep_start = timestep_start_obs + d). In code, d is `length_prefix_future_effective`.

Architecture
------------
- Actor thread: Executes actions at fps_observation, triggers inference when chunk runs low
- Inference thread: Runs policy with action prefix, produces pending chunk
- Shared state: Current observation, active/pending chunks, inference coordination

Usage:
    python -m so101_direct_manipulation.eval.eval_async_smooth \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --robot.id=arm_follower_0 \
        --policy.path=outputs/model/pretrained_model \
        --fps_policy=10 \
        --fps_interpolation=30 \
        --fps_observation=60 \
        --is_logging_rr=true
"""

import logging
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from threading import Event, Thread

import rerun as rr
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: F401
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun
from so101_direct_manipulation.eval.utils_eval import (
    SnapshotCameras,
    SnapshotMotors,
    StateBase,
    ThreadObsCameras,
    ThreadObsMotors,
    build_action_robot,
    build_obs_policy,
)
from so101_direct_manipulation.eval.utils_rerun import log_rr_action

# Import ACTSmooth for type registration and inference utilities
from lerobot_policy_act_smooth import ACTSmoothConfig  # noqa: F401
from lerobot_policy_act_smooth.modeling_act_smooth import INFERENCE_ACTION, INFERENCE_ACTION_IS_PAD

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalAsyncSmoothConfig:
    """Configuration for asynchronous policy evaluation with ACTSmooth."""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # Control parameters
    fps_policy: int = 10
    fps_observation: int = 60
    fps_interpolation: int | None = None  # None = fps_policy (no interpolation)
    duration_s_episode: float = 60.0

    # Override n_action_steps from policy config (None = use policy default)
    n_action_steps: int | None = None

    # Trigger inference when remaining actions <= threshold_remaining_actions.
    # When < length_prefix_future, fewer future actions are available at trigger time;
    # the model pads shorter prefixes, which are then masked out.
    threshold_remaining_actions: int | None = None  # None = length_prefix_future from policy

    # Injected delay (ms) to simulate extra inference latency
    delay_ms_injected: int = 0

    # Rerun
    is_logging_rr: bool = False
    is_logging_rr_cameras: bool = False
    path_recording_rr: str = ""

    def __post_init__(self):
        path_policy = parser.get_path_arg("policy")
        if path_policy:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(path_policy, cli_overrides=cli_overrides)
            self.policy.pretrained_path = path_policy

        if self.policy is None:
            raise ValueError("A policy must be provided via --policy.path=...")

        if self.path_recording_rr:
            Path(self.path_recording_rr).parent.mkdir(parents=True, exist_ok=True)

        if self.fps_interpolation is None:
            self.fps_interpolation = self.fps_policy

        if self.fps_observation < self.fps_interpolation:
            raise ValueError(
                f"fps_observation ({self.fps_observation}) must be >= fps_interpolation ({self.fps_interpolation})"
            )
        if self.fps_interpolation < self.fps_policy:
            raise ValueError(f"fps_interpolation ({self.fps_interpolation}) must be >= fps_policy ({self.fps_policy})")
        if self.fps_observation % self.fps_interpolation != 0:
            raise ValueError(
                f"fps_observation ({self.fps_observation}) must be a multiple of fps_interpolation ({self.fps_interpolation})"
            )
        if self.fps_interpolation % self.fps_policy != 0:
            raise ValueError(
                f"fps_interpolation ({self.fps_interpolation}) must be a multiple of fps_policy ({self.fps_policy})"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ============================================================================
# Action Chunk
# ============================================================================


@dataclass
class ActionChunk:
    """A chunk of actions with associated timing information for ACTSmooth.

    The policy produces a chunk of actions from an observation and an action prefix.
    The predicted action chunk corresponds to timesteps `[timestep_start, timestep_start + chunk_size)`.

    Timing (matches the training layout in configuration_act_smooth.py):

        Prefix = [t_{-k}, ..., t_{-1}, t_0, t_1, ..., t_{D-1}]
                 |------ k past ------|  |---- D committed ----|

        Target = [t_D, ..., t_{D+C-1}]   (= actions[0..C-1])

        timestep_start = timestep_start_obs + D

    Where D = length_prefix_future_effective:
        - min(length_prefix_future, actions available at trigger) for non-first chunks.
          When threshold_remaining_actions < length_prefix_future, fewer actions remain
          at trigger time. The model pads shorter prefixes.
        - 1 for the first chunk (no prior actions available)

    The future prefix starts at the observation timestep (t_0), so the first
    predicted action is at t_D = timestep_start_obs + D. No +1: t_0 is the
    action AT the observation timestep, already committed.

    Attributes:
        actions: Action tensor [n_actions, action_dim] as ABSOLUTE joint positions
        timestep_start_obs: Timestep when observation was captured for this chunk's inference
        length_prefix_future_effective: Number of future prefix actions used for this chunk's inference
        idx_chunk: Inference counter that generated this chunk (for logging)
    """

    actions: torch.Tensor
    timestep_start_obs: int  # When observation was captured
    length_prefix_future_effective: int  # Actual future prefix size (≤ length_prefix_future)
    idx_chunk: int

    @property
    def timestep_start(self) -> int:
        """Timestep when actions[0] executes."""
        return self.timestep_start_obs + self.length_prefix_future_effective

    def action_at(self, timestep: int) -> torch.Tensor | None:
        """Get action for the given timestep."""
        idx = timestep - self.timestep_start
        if 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None

    def cnt_actions_remaining_from(self, timestep: int) -> int:
        """Get number of actions from timestep onwards (including it).

        Inclusive so the inference trigger reads naturally:
            cnt_actions_remaining_from(t) <= threshold_remaining_actions
        means "the remaining actions (including the current one) are at or below
        the threshold", i.e. it's time to request a new chunk.
        """
        idx = timestep - self.timestep_start
        return max(0, len(self.actions) - idx)

    def actions_between(self, timestep_start: int, timestep_end: int) -> torch.Tensor:
        """Get actions for timesteps [timestep_start, timestep_end) as [1, n, action_dim]."""
        idx_start = max(0, timestep_start - self.timestep_start)
        idx_end = min(len(self.actions), timestep_end - self.timestep_start)
        return self.actions[idx_start:idx_end].unsqueeze(0)


# ============================================================================
# Shared State
# ============================================================================


@dataclass
class State(StateBase):
    """Centralized state for robot operations and thread communication."""

    timestep: int = 0

    action_chunk_active: ActionChunk | None = None
    action_chunk_pending: ActionChunk | None = None

    timestep_inference_requested: int | None = None
    snapshot_motors_inference_requested: SnapshotMotors | None = None
    snapshot_cameras_inference_requested: SnapshotCameras | None = None

    # Written by actor thread on exit, read by main thread after episode
    cnt_frames_observation: int = 0
    cnt_frames_interpolation: int = 0

    event_inference_requested: Event | None = None

    def __post_init__(self):
        super().__post_init__()  # creates lock and event_shutdown
        self.event_inference_requested = Event()


# ============================================================================
# Actor Thread — Pure Logic
# ============================================================================


@dataclass
class ArgsStepActorControlFrame:
    """Arguments for step_actor_control_frame."""

    timestep: int
    action_chunk_active: ActionChunk | None
    action_chunk_pending: ActionChunk | None
    is_inference_requested: bool
    threshold_remaining_actions: int


@dataclass
class OutputStepActorControlFrame:
    """Pure output of a single actor control frame decision."""

    action: torch.Tensor | None  # Action to send (raw, pre-postprocess)
    action_chunk_active: ActionChunk | None  # Updated active chunk (after potential switch)
    action_chunk_pending_consumed: bool  # Whether pending was consumed
    cnt_actions_remaining: int
    cnt_actions_discarded: int  # Actions discarded during switch
    should_request_inference: bool
    should_warn_interpolation_holding_action: bool  # Chunk exhausted, inference not ready
    action_interp_target: torch.Tensor | None  # Next action for interpolation endpoint


def step_actor_control_frame(args: ArgsStepActorControlFrame) -> OutputStepActorControlFrame:
    """Pure decision logic for one actor control frame.

    Handles chunk switching, action selection, inference triggering,
    and interpolation target computation — no I/O or state mutation.
    """
    action_chunk_active = args.action_chunk_active
    action_chunk_pending_consumed = False
    cnt_actions_discarded = 0

    # 1. Action selection: try active chunk, fall back to pending
    action = action_chunk_active.action_at(args.timestep) if action_chunk_active is not None else None
    if action is None and args.action_chunk_pending is not None:
        action = args.action_chunk_pending.action_at(args.timestep)
        if action is not None:
            cnt_actions_discarded = max(0, args.timestep - args.action_chunk_pending.timestep_start)
            action_chunk_active = args.action_chunk_pending
            action_chunk_pending_consumed = True

    has_unconsumed_pending = args.action_chunk_pending is not None and not action_chunk_pending_consumed
    can_request_inference = not args.is_inference_requested and not has_unconsumed_pending

    cnt_actions_remaining = (
        action_chunk_active.cnt_actions_remaining_from(args.timestep) if action_chunk_active is not None else 0
    )

    # 2. No action available — request inference and return early
    if action is None:
        return OutputStepActorControlFrame(
            action=None,
            action_chunk_active=action_chunk_active,
            action_chunk_pending_consumed=action_chunk_pending_consumed,
            cnt_actions_remaining=cnt_actions_remaining,
            cnt_actions_discarded=cnt_actions_discarded,
            should_request_inference=can_request_inference,
            should_warn_interpolation_holding_action=args.action_chunk_active is not None,
            action_interp_target=None,
        )

    # --- From here, action and action_chunk_active are guaranteed non-None ---

    # 3. Inference trigger: remaining actions (inclusive of current) are at or below
    #    the threshold — time for a new chunk.
    should_request_inference = can_request_inference and cnt_actions_remaining <= args.threshold_remaining_actions

    # 4. Interpolation target: next timestep's action
    timestep_next = args.timestep + 1
    action_next = action_chunk_active.action_at(timestep_next)
    if has_unconsumed_pending:
        action_next_from_pending = args.action_chunk_pending.action_at(timestep_next)
        if action_next_from_pending is not None:
            action_next = action_next_from_pending
    # No next action: hold last action as interpolation target (action gap)
    action_interp_target = action_next if action_next is not None else action.clone()
    should_warn_interpolation_holding_action = action_next is None

    return OutputStepActorControlFrame(
        action=action,
        action_chunk_active=action_chunk_active,
        action_chunk_pending_consumed=action_chunk_pending_consumed,
        cnt_actions_remaining=cnt_actions_remaining,
        cnt_actions_discarded=cnt_actions_discarded,
        should_request_inference=should_request_inference,
        should_warn_interpolation_holding_action=should_warn_interpolation_holding_action,
        action_interp_target=action_interp_target,
    )


# ============================================================================
# Actor Thread
# ============================================================================


def thread_actor_fn(
    robot,
    state: State,
    names_action: list[str],
    postprocessor: PolicyProcessorPipeline,
    cfg: EvalAsyncSmoothConfig,
    ts_episode_start: float,
) -> None:
    """Actor thread: executes actions from current chunk at target fps."""
    duration_s_frame_target = 1.0 / cfg.fps_observation
    cnt_frames_observation = 0
    cnt_frames_interpolation = 0
    action_chunk_active: ActionChunk | None = None

    cnt_frames_per_control_frame = cfg.fps_observation // cfg.fps_policy
    cnt_frames_per_interpolation_frame = cfg.fps_observation // cfg.fps_interpolation

    action_interpolation_start: torch.Tensor | None = None
    action_interpolation_end: torch.Tensor | None = None
    is_interpolation_ready: bool = False

    try:
        timestep = 0
        idx_frame = 0
        while not state.event_shutdown.is_set():
            ts_start_frame = time.perf_counter()

            is_control_frame = (idx_frame % cnt_frames_per_control_frame) == 0
            is_interpolation_frame = (idx_frame % cnt_frames_per_interpolation_frame) == 0

            # --- Control frame ---
            if is_control_frame:
                # Read snapshots under lock — see StateBase docstring.
                with state.lock:
                    snapshot_motors = state.snapshot_motors
                    snapshot_cameras = state.snapshot_cameras
                if snapshot_motors is None or snapshot_cameras is None:
                    # Observer threads not ready yet; wait one frame
                    time.sleep(duration_s_frame_target)
                    idx_frame += 1
                    cnt_frames_observation += 1
                    continue

                with state.lock:
                    state.timestep = timestep

                    output = step_actor_control_frame(
                        ArgsStepActorControlFrame(
                            timestep=timestep,
                            action_chunk_active=action_chunk_active,
                            action_chunk_pending=state.action_chunk_pending,
                            is_inference_requested=state.timestep_inference_requested is not None,
                            threshold_remaining_actions=cfg.threshold_remaining_actions,
                        )
                    )

                    action = output.action
                    action_chunk_active = output.action_chunk_active
                    action_chunk_pending_consumed = output.action_chunk_pending_consumed
                    cnt_actions_remaining = output.cnt_actions_remaining
                    cnt_actions_discarded = output.cnt_actions_discarded
                    should_request_inference = output.should_request_inference
                    should_warn_interpolation_holding_action = output.should_warn_interpolation_holding_action
                    action_interp_target = output.action_interp_target

                    if action_chunk_pending_consumed:
                        state.action_chunk_active = action_chunk_active
                        state.action_chunk_pending = None
                        logging.info(
                            f"[ACTOR] Switched to idx_chunk={action_chunk_active.idx_chunk} "
                            f"(timestep_start_obs={action_chunk_active.timestep_start_obs}, "
                            f"length_prefix_future_effective={action_chunk_active.length_prefix_future_effective}, "
                            f"cnt_actions_discarded={cnt_actions_discarded})"
                        )

                    if should_request_inference:
                        state.timestep_inference_requested = timestep
                        state.snapshot_motors_inference_requested = snapshot_motors
                        state.snapshot_cameras_inference_requested = snapshot_cameras
                        state.event_inference_requested.set()

                    idx_chunk = action_chunk_active.idx_chunk if action_chunk_active else -1

                if should_warn_interpolation_holding_action and cfg.fps_interpolation > cfg.fps_policy:
                    if action is None:
                        logging.warning(
                            f"[ACTOR] Action gap at timestep={timestep}: active chunk exhausted, "
                            f"inference not ready. Holding last action."
                        )
                    else:
                        logging.warning(
                            f"[ACTOR] Interpolation holding at timestep={timestep}: "
                            f"no next action available, using current action as interpolation target."
                        )

                if action is not None:
                    action = postprocessor(action.unsqueeze(0))  # Unnormalize
                    action = action.squeeze(0).cpu()
                    action_robot = build_action_robot(action, names_action)
                    # robot.send_action is motor bus I/O — must be under lock.
                    # See StateBase docstring for the locking invariant.
                    with state.lock:
                        robot.send_action(action_robot)

                    if cfg.is_logging_rr:
                        log_rr_action(
                            ts=ts_start_frame - ts_episode_start,
                            timestep=timestep + 1,  # matches post-increment value logged below
                            idx_chunk=idx_chunk,
                            value=action_robot,
                        )

                    action_interpolation_start = action.clone()
                    if action_interp_target is not None:
                        action_interpolation_end = postprocessor(action_interp_target.unsqueeze(0)).squeeze(0).cpu()
                    else:
                        action_interpolation_end = action_interpolation_start.clone()
                    is_interpolation_ready = True

                logging.info(
                    f"[ACTOR] timestep={timestep} | chunk={idx_chunk} | actions_remaining={cnt_actions_remaining}"
                )

                timestep += 1

            # --- Interpolation frame ---
            # Motor observer handles motor reads; actor only sends interpolated action.
            if not is_control_frame and is_interpolation_frame:
                if is_interpolation_ready and action_interpolation_start is not None:
                    idx_within_period = idx_frame % cnt_frames_per_control_frame
                    t = idx_within_period / cnt_frames_per_control_frame

                    action_interpolation = (1.0 - t) * action_interpolation_start + t * action_interpolation_end
                    action_robot_interp = build_action_robot(action_interpolation, names_action)
                    with state.lock:
                        robot.send_action(action_robot_interp)
                    cnt_frames_interpolation += 1

                    if cfg.is_logging_rr:
                        log_rr_action(
                            ts=ts_start_frame - ts_episode_start,
                            timestep=timestep,
                            idx_chunk=idx_chunk,
                            value=action_robot_interp,
                        )

            # Observation-only frames: motor observer handles motor reads, nothing to do here.

            cnt_frames_observation += 1
            idx_frame += 1

            duration_s_frame = time.perf_counter() - ts_start_frame
            precise_sleep(max(0, (duration_s_frame_target - duration_s_frame) - 0.001))

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        state.event_shutdown.set()
        sys.exit(1)

    state.cnt_frames_observation = cnt_frames_observation
    state.cnt_frames_interpolation = cnt_frames_interpolation
    state.timestep = timestep
    logging.info(f"[ACTOR] Thread shutting down. timestep={timestep}")


# ============================================================================
# Inference Thread — Pure Logic
# ============================================================================


@dataclass
class ArgsMakePrefixForInference:
    """Arguments for make_prefix_for_inference."""

    timestep: int
    action_chunk_active: ActionChunk
    length_prefix_future: int
    length_prefix_past: int


@dataclass
class OutputMakePrefixForInference:
    """Pure output of inference pre-processing (prefix extraction)."""

    action_prefix_future: torch.Tensor  # [1, D, action_dim]
    action_prefix_past: torch.Tensor  # [1, k, action_dim]
    length_prefix_future_effective: int


def make_prefix_for_inference(args: ArgsMakePrefixForInference) -> OutputMakePrefixForInference:
    """Build action prefix from active chunk. Non-first chunks only.

    `timestep` is the observation timestep (t_0) for this inference call.
    Since actions_between is [start, end), the ranges align with the training layout
    from configuration_act_smooth.py:

        past:   [timestep - k, timestep)            -> t_{-k}, ..., t_{-1}
        future: [timestep, timestep + D)             -> t_0, t_1, ..., t_{D-1}

    Requests up to length_prefix_future actions, but actions_between clips to
    what's available. When threshold_remaining_actions < length_prefix_future,
    fewer actions remain at trigger time, so length_prefix_future_effective
    = action_prefix_future.shape[1] <= length_prefix_future.
    """
    action_prefix_future = args.action_chunk_active.actions_between(
        args.timestep, args.timestep + args.length_prefix_future
    )
    action_prefix_past = args.action_chunk_active.actions_between(
        args.timestep - args.length_prefix_past, args.timestep
    )
    return OutputMakePrefixForInference(
        action_prefix_future=action_prefix_future,
        action_prefix_past=action_prefix_past,
        length_prefix_future_effective=action_prefix_future.shape[1],
    )


@dataclass
class ArgsMakePrefixForInferenceFirst:
    """Arguments for make_prefix_for_inference_first."""

    snapshot_motors: SnapshotMotors
    names_motor: list[str]
    device: torch.device
    action_dim: int
    preprocessor: PolicyProcessorPipeline


def make_prefix_for_inference_first(args: ArgsMakePrefixForInferenceFirst) -> OutputMakePrefixForInference:
    """Build action prefix for the first inference (no prior chunk).

    Uses current robot state as 1-action future prefix.
    """
    current_state = torch.tensor(
        [args.snapshot_motors.positions[name] for name in args.names_motor], device=args.device, dtype=torch.float32
    )
    action_prefix_future = current_state.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
    for step in args.preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            action_prefix_future = step._normalize_action(action_prefix_future, inverse=False)
            break
    action_prefix_past = torch.zeros(1, 0, args.action_dim, device=args.device)
    return OutputMakePrefixForInference(
        action_prefix_future=action_prefix_future,
        action_prefix_past=action_prefix_past,
        length_prefix_future_effective=1,
    )


# ============================================================================
# Inference Thread
# ============================================================================


def thread_inference_fn(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    state: State,
    device: torch.device,
    cfg: EvalAsyncSmoothConfig,
    n_action_steps: int,
    names_motor: list[str],
    names_camera: list[str],
    robot_type: str,
    length_prefix_future: int,
    length_prefix_past: int,
) -> None:
    """Inference thread: waits for signal, runs inference with action prefix, creates new action chunk."""
    idx_chunk = 0

    try:
        while not state.event_shutdown.is_set():
            if not state.event_inference_requested.wait(timeout=1.0):
                continue
            state.event_inference_requested.clear()

            with state.lock:
                timestep = state.timestep_inference_requested
                snapshot_motors = state.snapshot_motors_inference_requested
                snapshot_cameras = state.snapshot_cameras_inference_requested
                action_chunk_active = state.action_chunk_active

                if timestep is None or snapshot_motors is None:
                    state.timestep_inference_requested = None
                    state.snapshot_motors_inference_requested = None
                    state.snapshot_cameras_inference_requested = None
                    continue

            # Pre-processing: build action prefix
            if action_chunk_active is not None:
                output_make_prefix_for_inference = make_prefix_for_inference(
                    ArgsMakePrefixForInference(
                        timestep=timestep,
                        action_chunk_active=action_chunk_active,
                        length_prefix_future=length_prefix_future,
                        length_prefix_past=length_prefix_past,
                    )
                )
            else:
                output_make_prefix_for_inference = make_prefix_for_inference_first(
                    ArgsMakePrefixForInferenceFirst(
                        snapshot_motors=snapshot_motors,
                        names_motor=names_motor,
                        device=device,
                        action_dim=policy.config.action_feature.shape[0],
                        preprocessor=preprocessor,
                    )
                )

            action_prefix_future = output_make_prefix_for_inference.action_prefix_future
            action_prefix_past = output_make_prefix_for_inference.action_prefix_past
            length_prefix_future_effective = output_make_prefix_for_inference.length_prefix_future_effective

            assert action_prefix_future.shape[1] == length_prefix_future_effective, (
                f"Expected {length_prefix_future_effective} future prefix actions, got {action_prefix_future.shape[1]}"
            )
            if action_chunk_active is not None:
                assert action_prefix_past.shape[1] == length_prefix_past, (
                    f"Expected {length_prefix_past} past prefix actions, got {action_prefix_past.shape[1]}"
                )

            if state.event_shutdown.is_set():
                break

            # Build observation frame with state AND images (must go through preprocessing)
            obs_policy = build_obs_policy(
                positions=snapshot_motors.positions,
                images=snapshot_cameras.images,
                names_motor=names_motor,
                names_camera=names_camera,
            )

            logging.info(
                f"[INFERENCE] Starting idx_chunk={idx_chunk + 1} | "
                f"timestep={timestep} | length_prefix_future_effective={length_prefix_future_effective}"
            )
            ts_start_inference = time.perf_counter()

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type)
                if device.type == "cuda" and policy.config.use_amp
                else nullcontext(),
            ):
                obs_policy = prepare_observation_for_inference(obs_policy, device, None, robot_type)
                obs_policy = preprocessor(obs_policy)

                # Build inference action prefix tensors
                inference_action, inference_action_is_pad = policy.build_inference_action_prefix(
                    action_prefix_future=action_prefix_future,
                    action_prefix_past=action_prefix_past,
                )
                obs_policy[INFERENCE_ACTION] = inference_action
                obs_policy[INFERENCE_ACTION_IS_PAD] = inference_action_is_pad

                chunk_policy = policy.predict_action_chunk(obs_policy)
                chunk_policy = chunk_policy[:, :n_action_steps, :]

            if cfg.delay_ms_injected > 0:
                time.sleep(cfg.delay_ms_injected / 1000.0)

            duration_ms_inference = (time.perf_counter() - ts_start_inference) * 1000

            if state.event_shutdown.is_set():
                break

            idx_chunk += 1

            # Create action chunk from model output
            # timestep_start = timestep + length_prefix_future_effective
            action_chunk_pending = ActionChunk(
                actions=chunk_policy.squeeze(0),
                timestep_start_obs=timestep,
                length_prefix_future_effective=length_prefix_future_effective,
                idx_chunk=idx_chunk,
            )

            with state.lock:
                state.action_chunk_pending = action_chunk_pending
                state.timestep_inference_requested = None
                state.snapshot_motors_inference_requested = None
                state.snapshot_cameras_inference_requested = None

            logging.info(
                f"[INFERENCE] idx_chunk={idx_chunk} | duration_ms={duration_ms_inference:.1f}ms | "
                f"timestep={timestep} | length_prefix_future_effective={action_chunk_pending.length_prefix_future_effective} | "
                f"timestep_start={action_chunk_pending.timestep_start}"
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
def main(cfg: EvalAsyncSmoothConfig) -> None:
    """Main entry point for asynchronous policy evaluation with action prefix."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.is_logging_rr:
        init_rerun(session_name="eval_async_smooth")
        if cfg.path_recording_rr:
            rr.save(cfg.path_recording_rr)
            logging.info(f"Recording to {cfg.path_recording_rr}")

    device = get_safe_torch_device(cfg.policy.device if cfg.policy.device else "auto")
    logging.info(f"Using device: {device}")

    logging.info("Creating robot...")
    robot = make_robot_from_config(cfg.robot)

    logging.info(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path)
    policy.to(device)

    length_prefix_future = getattr(policy.config, "length_prefix_future", 0)
    length_prefix_past = getattr(policy.config, "length_prefix_past", 0)
    if length_prefix_future < 1:
        raise ValueError(
            f"eval_async_smooth requires ACTSmooth policy with length_prefix_future >= 1, got {length_prefix_future}"
        )

    # Resolve threshold_remaining_actions: default to length_prefix_future
    if cfg.threshold_remaining_actions is None:
        cfg.threshold_remaining_actions = length_prefix_future

    logging.info(
        f"Policy: length_prefix_future={length_prefix_future}, length_prefix_past={length_prefix_past}. "
        f"threshold_remaining_actions={cfg.threshold_remaining_actions}"
    )

    # TODO: support threshold_remaining_actions > length_prefix_future.
    # When threshold > length_prefix_future, the actor should switch to the pending chunk as soon
    # as the future prefix period ends (eager switching), discarding the remaining
    # (threshold - length_prefix_future) actions from the active chunk. The current lazy-switching
    # logic instead keeps executing the active chunk until exhaustion, which runs actions that
    # should have been dropped and feeds an incorrect past prefix to the next inference call.
    if cfg.threshold_remaining_actions > length_prefix_future:
        raise ValueError(
            f"threshold_remaining_actions ({cfg.threshold_remaining_actions}) must be <= "
            f"length_prefix_future ({length_prefix_future}). "
            f"Larger values are not yet supported: the actor would continue executing the active "
            f"chunk past the committed future prefix window instead of switching to the pending chunk."
        )

    # The future prefix starts at t_0 (observation timestep, already executing), so only d-1
    # actions absorb inference latency. With d=1 there's no interpolation target at chunk
    # switches while inference runs, causing the actor to hold the last action.
    is_interpolation_active = cfg.fps_interpolation > cfg.fps_policy
    if is_interpolation_active and cfg.threshold_remaining_actions < 2:
        logging.warning(
            f"Interpolation is active (fps_interpolation={cfg.fps_interpolation} > fps_policy={cfg.fps_policy}) "
            f"but threshold_remaining_actions={cfg.threshold_remaining_actions} < 2. If inference latency exceeds "
            f"{cfg.threshold_remaining_actions - 1} policy steps, there will be no interpolation target at chunk "
            f"switches, causing the actor to hold the last action."
        )

    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    thread_inference = None
    thread_actor = None
    thread_obs_motors = None
    thread_obs_cameras = None
    ts_episode_start = None
    state = None

    try:
        logging.info("Connecting to robot...")
        robot.connect()

        logging.info(
            f"Starting episode (max {cfg.duration_s_episode}s at {cfg.fps_policy} fps policy, "
            f"{cfg.fps_interpolation} fps interpolation, {cfg.fps_observation} fps observation)"
        )
        logging.info(f"Inference triggers when actions_remaining <= {cfg.threshold_remaining_actions}")

        policy.reset()
        preprocessor.reset()

        state = State()

        names_action = list(robot.action_features.keys())
        names_motor = [k for k in robot.observation_features if robot.observation_features[k] is float]
        names_camera = [k for k in robot.observation_features if isinstance(robot.observation_features[k], tuple)]

        robot_type = robot.robot_type

        n_action_steps_policy = getattr(policy.config, "n_action_steps", 1)
        n_action_steps = cfg.n_action_steps if cfg.n_action_steps is not None else n_action_steps_policy
        logging.info(f"Using n_action_steps: {n_action_steps}")

        # Ensure chunk is long enough to read past actions when inference triggers.
        # Inference triggers when cnt_actions_remaining <= threshold_remaining_actions,
        # i.e. when remaining actions (including current) are at or below the threshold.
        # At trigger, current index = n_action_steps - threshold_remaining_actions.
        # We need current index >= length_prefix_past for the past prefix.
        # Therefore: n_action_steps >= threshold_remaining_actions + length_prefix_past
        min_n_action_steps = cfg.threshold_remaining_actions + length_prefix_past
        if n_action_steps < min_n_action_steps:
            raise ValueError(
                f"n_action_steps ({n_action_steps}) must be >= threshold_remaining_actions ({cfg.threshold_remaining_actions}) "
                f"+ length_prefix_past ({length_prefix_past}) = {min_n_action_steps}"
            )

        # === WARMUP INFERENCE ===
        # Run one inference pass to trigger JIT compilation and CUDA kernel loading.
        # Done before observer threads start so robot.get_observation() is safe to call.
        logging.info("Running warmup inference...")
        obs_robot_warmup = robot.get_observation()
        obs_policy_warmup = build_obs_policy(
            positions={name: obs_robot_warmup[name] for name in names_motor},
            images={name: obs_robot_warmup[name] for name in names_camera},
            names_motor=names_motor,
            names_camera=names_camera,
        )

        ts_start_warmup = time.perf_counter()
        use_amp = policy.config.use_amp
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
        ):
            obs_policy_warmup = prepare_observation_for_inference(obs_policy_warmup, device, None, robot_type)
            obs_policy_warmup = preprocessor(obs_policy_warmup)

            # Build inference action prefix tensors for warmup.
            # WARNING: The model requires length_prefix_future_effective >= 1 (the "delay").
            # Model was trained on delays {1..length_prefix_future}, so delay=0 is out-of-distribution.
            # Use the current robot state as a 1-action prefix.
            action_dim = policy.config.action_feature.shape[0]
            array_proprio_warmup = [obs_robot_warmup[name] for name in names_motor]
            current_state_warmup = torch.tensor(array_proprio_warmup, device=device, dtype=torch.float32)
            action_prefix_future_warmup = current_state_warmup.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            for step in preprocessor.steps:
                if isinstance(step, NormalizerProcessorStep):
                    action_prefix_future_warmup = step._normalize_action(action_prefix_future_warmup, inverse=False)
                    break
            inference_action, inference_action_is_pad = policy.build_inference_action_prefix(
                action_prefix_future=action_prefix_future_warmup,
                action_prefix_past=torch.zeros(1, 0, action_dim, device=device),
            )
            obs_policy_warmup[INFERENCE_ACTION] = inference_action
            obs_policy_warmup[INFERENCE_ACTION_IS_PAD] = inference_action_is_pad

            _ = policy.predict_action_chunk(obs_policy_warmup)
        duration_ms_warmup = (time.perf_counter() - ts_start_warmup) * 1000
        logging.info(f"Warmup inference: {duration_ms_warmup:.1f}ms")

        # Reset preprocessor after warmup (it may have internal state)
        preprocessor.reset()

        ts_episode_start = time.perf_counter()

        # Start observer threads before actor and inference threads.
        # They use state.event_shutdown so they stop when the episode ends.
        thread_obs_motors = ThreadObsMotors(
            bus=robot.bus,
            fps_obs=cfg.fps_observation,
            ts_episode_start=ts_episode_start,
            state=state,
            is_logging_rr=cfg.is_logging_rr,
        )
        thread_obs_cameras = ThreadObsCameras(
            cameras=robot.cameras,
            ts_episode_start=ts_episode_start,
            state=state,
            is_logging_rr=cfg.is_logging_rr and cfg.is_logging_rr_cameras,
        )

        thread_inference = Thread(
            target=thread_inference_fn,
            args=(
                policy,
                preprocessor,
                state,
                device,
                cfg,
                n_action_steps,
                names_motor,
                names_camera,
                robot_type,
                length_prefix_future,
                length_prefix_past,
            ),
            daemon=True,
            name="Inference",
        )

        thread_actor = Thread(
            target=thread_actor_fn,
            args=(
                robot,
                state,
                names_action,
                postprocessor,
                cfg,
                ts_episode_start,
            ),
            daemon=True,
            name="Actor",
        )

        thread_obs_motors.start()
        thread_obs_cameras.start()
        thread_inference.start()
        thread_actor.start()

        while not state.event_shutdown.is_set():
            duration_s_episode = time.perf_counter() - ts_episode_start

            if duration_s_episode >= cfg.duration_s_episode:
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        if state is not None:
            state.event_shutdown.set()
            state.event_inference_requested.set()

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

        if thread_obs_motors is not None and thread_obs_motors.is_alive():
            thread_obs_motors.join(timeout=2.0)
        if thread_obs_cameras is not None and thread_obs_cameras.is_alive():
            thread_obs_cameras.join(timeout=2.0)

        if ts_episode_start is not None and state is not None:
            duration_s_episode = time.perf_counter() - ts_episode_start
            logging.info(f"Episode completed in {duration_s_episode:.1f}s")

            if duration_s_episode > 0:
                fps_observation_real = state.cnt_frames_observation / duration_s_episode
                fps_interpolation_real = (state.cnt_frames_interpolation + state.timestep) / duration_s_episode
                fps_policy_real = state.timestep / duration_s_episode
                logging.info(
                    f"Real FPS: observation={fps_observation_real:.1f} (target={cfg.fps_observation}), "
                    f"interpolation={fps_interpolation_real:.1f} (target={cfg.fps_interpolation}), "
                    f"policy={fps_policy_real:.1f} (target={cfg.fps_policy})"
                )

        # Run homing sequence to safely park the arm before disconnecting
        if robot and robot.is_connected:
            logging.info("Running homing sequence...")
            try:
                from so101_direct_manipulation.shared.homing import run_homing_sequence

                run_homing_sequence(robot, enable_rerun_logging=False)
            except Exception as e:
                logging.error(f"Homing failed: {e}")

            logging.info("Disconnecting robot...")
            robot.disconnect()


if __name__ == "__main__":
    main()
