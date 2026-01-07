#!/usr/bin/env python
"""Synchronous ACT Relative RTC policy evaluation with action discarding (UMI-style).

This script runs policy inference synchronously but discards the first actions in
each chunk that correspond to timesteps already elapsed due to inference latency.
This matches the approach described in the UMI paper.

The policy "assumes" the first action in a chunk should be executed immediately with
the observation that produced it, but inference latency makes this impossible. By
discarding elapsed actions, we better align predicted actions with actual execution.

The inference logic is fully decoupled from the policy's internal queue mechanism.
We call predict_action_chunk() directly and handle:
- Observation delta computation (obs[t] - obs[t-N])
- Delta observation normalization
- Relative action unnormalization
- Relative-to-absolute action conversion

Usage:
    python -m so101_data_collection.eval.eval_sync_discard \
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
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pprint import pformat

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
class EvalDiscardConfig:
    """Configuration for synchronous policy evaluation with action discarding."""

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
# Observation Queue Management
# ============================================================================


class ObservationQueue:
    """Manages observation history for delta computation.

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

    def update(self, obs_state: torch.Tensor) -> None:
        """Add new observation to queue.

        On first call, fills queue with copies of the observation.
        """
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
        obs_list = list(self.queue)
        obs_state_t_minus_n = obs_list[0]  # oldest in queue
        obs_state_t = obs_list[-1]  # current observation
        delta_obs = obs_state_t - obs_state_t_minus_n
        return delta_obs, obs_state_t

    def reset(self) -> None:
        """Clear the queue for a new episode."""
        self.queue.clear()


# ============================================================================
# Inference Helpers
# ============================================================================


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
# Main Evaluation Loop
# ============================================================================


def run_episode_sync_discard(
    robot,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    events: dict,
    cfg: EvalDiscardConfig,
    device: torch.device,
    latency_tracker: LatencyTracker,
    discard_tracker: DiscardTracker,
    ds_meta,
) -> None:
    """Run a single episode with synchronous inference and action discarding.

    Control flow:
    1. Get observation
    2. Run inference to get full action chunk (n_action_steps actions)
    3. Compute n_skip based on inference + execution latency
    4. Execute actions[n_skip:] at target fps
    5. Repeat until episode ends

    The inference logic is fully decoupled from the policy's internal queue.
    We call predict_action_chunk() directly and handle all conversions here.
    """
    logging.info(f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps} fps)")
    logging.info(f"Execution latency: {cfg.execution_latency_ms}ms")
    logging.info("Press Right arrow to exit early, Esc to stop completely")

    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    latency_tracker.reset()
    discard_tracker.reset()

    # Create observation queue for delta computation
    obs_state_delta_frames = getattr(policy.config, "obs_state_delta_frames", 1)
    obs_queue = ObservationQueue(obs_state_delta_frames)

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
    logging.info(
        f"Policy n_action_steps: {policy_n_action_steps}, using: {n_action_steps}"
    )
    logging.info(f"Policy obs_state_delta_frames: {obs_state_delta_frames}")
    logging.info(f"Policy has_relative_stats: {policy.has_relative_stats}")

    # Control loop timing
    dt_target = 1.0 / cfg.fps
    start_episode_t = time.perf_counter()
    chunk_idx = 0
    total_actions_executed = 0
    total_actions_discarded = 0

    while True:
        elapsed = time.perf_counter() - start_episode_t

        # Check termination conditions
        if events.get("exit_early") or events.get("stop_recording"):
            events["exit_early"] = False
            break
        if elapsed >= cfg.episode_time_s:
            logging.info("Episode time limit reached")
            break

        # === INFERENCE PHASE ===
        # Get observation
        raw_obs = robot.get_observation()

        # Build observation frame directly (bypassing build_dataset_frame)
        # This avoids issues when robot cameras don't match training dataset cameras
        observation_frame = {}

        # observation.state: array of joint positions
        state_values = [raw_obs[motor_name] for motor_name in motor_names]
        observation_frame["observation.state"] = np.array(
            state_values, dtype=np.float32
        )

        # observation.images.X: camera images
        for cam_name in camera_names:
            observation_frame[f"observation.images.{cam_name}"] = raw_obs[cam_name]

        # Run inference to get full action chunk
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

        # Record latency
        latency_tracker.record(inference_ms, log_to_rerun=cfg.display_data)

        # === DISCARD COMPUTATION ===
        # Compute how many actions to skip based on latency
        n_skip = compute_actions_to_skip(
            inference_ms=inference_ms,
            execution_latency_ms=cfg.execution_latency_ms,
            fps=cfg.fps,
        )

        # Clamp n_skip to leave at least 1 action to execute
        n_skip = min(n_skip, n_action_steps - 1)

        # Record discarded count
        discard_tracker.record(n_skip, log_to_rerun=cfg.display_data)
        total_actions_discarded += n_skip

        # Get actions to execute (skip first n_skip)
        # action_chunk shape: [batch, n_action_steps, action_dim]
        actions_to_execute = action_chunk[:, n_skip:, :]
        n_actions = actions_to_execute.shape[1]

        if chunk_idx % 10 == 0:
            logging.info(
                f"Chunk {chunk_idx}: inference={inference_ms:.1f}ms, "
                f"skip={n_skip}, execute={n_actions}"
            )

        # === EXECUTION PHASE ===
        # Execute remaining actions at target fps
        for action_idx in range(n_actions):
            action_start_t = time.perf_counter()

            # Check for early exit
            if events.get("exit_early") or events.get("stop_recording"):
                break

            # Check time limit
            if time.perf_counter() - start_episode_t >= cfg.episode_time_s:
                break

            # Get action for this step
            # actions_to_execute shape: [batch, n_actions, action_dim]
            action_tensor = actions_to_execute[:, action_idx, :]  # [batch, action_dim]

            # Move to CPU for robot action conversion
            action_tensor = action_tensor.cpu()

            # Convert action tensor to robot action dict
            # make_robot_action uses ds_features[ACTION]["names"] to build keys
            # If dataset action names include '.pos' suffix, robot_action is ready to use
            robot_action = make_robot_action(action_tensor, ds_features)

            # Log to rerun
            if cfg.display_data:
                # Get fresh observation for visualization
                vis_obs = robot.get_observation()
                log_rerun_data(observation=vis_obs, action=robot_action)

            # Send action to robot
            robot.send_action(robot_action)

            total_actions_executed += 1

            # Sleep to maintain target fps
            action_duration = time.perf_counter() - action_start_t
            sleep_time = dt_target - action_duration
            if sleep_time > 0:
                precise_sleep(sleep_time)

        chunk_idx += 1

    # Log final stats
    total_time = time.perf_counter() - start_episode_t
    actual_fps = total_actions_executed / total_time if total_time > 0 else 0
    logging.info(
        f"Episode complete: {total_actions_executed} actions in {total_time:.1f}s "
        f"({actual_fps:.1f} fps), {total_actions_discarded} actions discarded"
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
def main(cfg: EvalDiscardConfig) -> None:
    """Main entry point for synchronous policy evaluation with discarding."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Initialize rerun if displaying data
    if cfg.display_data:
        init_rerun(session_name="eval_sync_discard")

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

    try:
        # Connect robot
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        # Run episode
        log_say("Starting policy evaluation with action discarding", cfg.play_sounds)
        run_episode_sync_discard(
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
        # Cleanup
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()

        if not is_headless() and listener:
            listener.stop()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
