#!/usr/bin/env python
"""
Measure inference latency for a policy.

This script benchmarks policy inference time using synthetic observations.
Supports CPU, CUDA, and MPS devices (auto-detected or manually specified).

The inference pipeline matches eval_sync.py:
1. Prepare observation for inference (convert to tensors, add batch dim)
2. Apply preprocessor (image normalization, device placement)
3. Update observation queue and compute delta
4. Normalize delta observation (if policy has relative stats)
5. Call policy.predict_action_chunk() to get normalized relative actions
6. Unnormalize relative actions (if policy has relative stats)
7. Convert relative to absolute: absolute = relative + obs[t]

Usage:
    python -m so101_data_collection.latency.measure_inference \
        --policy.path=outputs/cube_hand_guided_act_umi_wrist_7_16k/pretrained_model_migrated \
        --dataset_repo_id=giacomoran/cube_hand_guided \
        --iterations=1000 \
        --warmup=10
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device

# Import custom policy configs so draccus can find them when loading configs
try:
    from lerobot_policy_act_relative_rtc import ACTRelativeRTCConfig  # noqa: F401
except ImportError:
    pass  # Policy config not available, will fail later if needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    """Statistics for inference latency measurements."""

    latencies_ms: list[float] = field(default_factory=list)

    def add(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        """Compute summary statistics."""
        if not self.latencies_ms:
            return {"count": 0}

        arr = np.array(self.latencies_ms)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


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
        self.queue: list[torch.Tensor] = []

    def update(self, obs_state: torch.Tensor) -> None:
        """Add new observation to queue.

        On first call, fills queue with copies of the observation.
        """
        if len(self.queue) < self.obs_state_delta_frames + 1:
            # Initialize by copying first observation until queue is full
            while len(self.queue) < self.obs_state_delta_frames + 1:
                self.queue.append(obs_state.clone())
        else:
            self.queue.append(obs_state)
            # Keep only last (delta_frames + 1) observations
            self.queue = self.queue[-(self.obs_state_delta_frames + 1) :]

    def get_delta(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute delta observation.

        Returns:
            Tuple of (delta_obs, obs_state_t) where:
            - delta_obs = obs[t] - obs[t - obs_state_delta_frames]
            - obs_state_t = current observation (needed for absolute action conversion)
        """
        obs_state_t_minus_n = self.queue[0]  # oldest in queue
        obs_state_t = self.queue[-1]  # current observation
        delta_obs = obs_state_t - obs_state_t_minus_n
        return delta_obs, obs_state_t

    def reset(self) -> None:
        """Clear the queue for a new run."""
        self.queue.clear()


def create_synthetic_observation(
    state_dim: int,
    camera_names: list[str],
    image_shape: tuple[int, int, int] = (480, 640, 3),
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Create a synthetic observation frame for benchmarking.

    Args:
        state_dim: Dimension of state vector
        camera_names: List of camera names (e.g., ["wrist", "top"])
        image_shape: Shape of images (height, width, channels)
        device: Optional device hint (not used, kept for API consistency)

    Returns:
        Observation dict with numpy arrays
    """
    observation = {}

    # Generate random state values in reasonable range [-1, 1]
    observation["observation.state"] = np.random.uniform(
        -1.0, 1.0, size=(state_dim,)
    ).astype(np.float32)

    # Generate random images (uint8)
    for cam_name in camera_names:
        key = f"observation.images.{cam_name}"
        observation[key] = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)

    return observation


def run_inference_chunk(
    observation_frame: dict[str, np.ndarray],
    obs_queue: ObservationQueue,
    policy,
    device: torch.device,
    preprocessor,
    n_action_steps: int,
    use_amp: bool,
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
        policy: The policy (PreTrainedPolicy)
        device: Torch device for inference
        preprocessor: Pipeline for observation preprocessing
        n_action_steps: Number of actions to return from the chunk
        use_amp: Whether to use automatic mixed precision
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
            observation_frame, device, task=None, robot_type=robot_type
        )
        observation = preprocessor(observation)

        # Ensure all observations are on the correct device (preprocessor might use config device)
        observation = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in observation.items()
        }

        # Step 3: Update observation queue and compute delta
        obs_state = observation["observation.state"]  # [batch, state_dim]
        obs_queue.update(obs_state)
        delta_obs, obs_state_t = obs_queue.get_delta()

        # Step 4: Normalize delta observation if stats are available
        if hasattr(policy, "has_relative_stats") and policy.has_relative_stats:
            # Debug: Check device mismatch before calling normalizer
            if delta_obs.device != policy.delta_obs_normalizer.mean.device:
                logger.error(
                    f"Device mismatch! delta_obs on {delta_obs.device}, "
                    f"normalizer mean on {policy.delta_obs_normalizer.mean.device}"
                )
                # Move delta_obs to match normalizer device
                delta_obs = delta_obs.to(policy.delta_obs_normalizer.mean.device)
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
        if hasattr(policy, "has_relative_stats") and policy.has_relative_stats:
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


def run_benchmark(
    policy_path: str,
    dataset_repo_id: str,
    device_str: str | None = None,
    iterations: int = 1000,
    warmup: int = 10,
    n_action_steps: int | None = None,
) -> InferenceStats:
    """Run inference benchmark.

    Args:
        policy_path: Path to policy directory
        dataset_repo_id: Dataset repository ID (required for policy loading)
        device_str: Device string ("cpu", "cuda", "mps", or None for auto-detect)
        iterations: Number of inference iterations to run
        warmup: Number of warmup iterations (not counted in stats)
        n_action_steps: Override n_action_steps from policy config

    Returns:
        InferenceStats with latency measurements
    """
    logger.info("=" * 80)
    logger.info("Policy Inference Latency Benchmark")
    logger.info("=" * 80)
    logger.info(f"Policy path: {policy_path}")
    logger.info(f"Dataset repo ID: {dataset_repo_id}")
    logger.info(f"Iterations: {iterations} (warmup: {warmup})")

    # Auto-select device if not specified
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
        logger.info(f"Auto-selected device: {device_str}")
    else:
        logger.info(f"Using device: {device_str}")

    device = get_safe_torch_device(device_str)
    logger.info(f"Device object: {device}")

    # Load dataset metadata (required for policy loading)
    logger.info(f"Loading dataset metadata from {dataset_repo_id}...")
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)

    # Create policy and processors
    logger.info(f"Loading policy from {policy_path}...")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device_str

    policy = make_policy(policy_cfg, ds_meta=ds_meta, env_cfg=None)

    # Explicitly move policy to the correct device
    # Use _apply to recursively move all parameters and buffers
    policy._apply(lambda t: t.to(device))

    # Also move normalizers if they exist (they're not registered as submodules)
    # Use _apply to recursively move all parameters and buffers
    if hasattr(policy, "delta_obs_normalizer"):
        policy.delta_obs_normalizer._apply(lambda t: t.to(device))
        logger.info(
            f"delta_obs_normalizer mean device: {policy.delta_obs_normalizer.mean.device}, "
            f"std device: {policy.delta_obs_normalizer.std.device}"
        )
    if hasattr(policy, "relative_action_normalizer"):
        policy.relative_action_normalizer._apply(lambda t: t.to(device))
        logger.info(
            f"relative_action_normalizer mean device: {policy.relative_action_normalizer.mean.device}, "
            f"std device: {policy.relative_action_normalizer.std.device}"
        )

    logger.info(f"Policy moved to device: {device}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
    )

    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Get policy config values
    policy_n_action_steps = getattr(policy.config, "n_action_steps", 1)
    n_action_steps = (
        n_action_steps if n_action_steps is not None else policy_n_action_steps
    )
    obs_state_delta_frames = getattr(policy.config, "obs_state_delta_frames", 1)
    use_amp = getattr(policy.config, "use_amp", False)

    logger.info(
        f"Policy n_action_steps: {policy_n_action_steps}, using: {n_action_steps}"
    )
    logger.info(f"Policy obs_state_delta_frames: {obs_state_delta_frames}")
    logger.info(f"Policy use_amp: {use_amp}")

    # Get observation features from dataset
    ds_features = ds_meta.features

    # Extract state dimension from feature dict
    # ds_features["observation.state"] is a dict with "shape" key
    state_feature = ds_features.get("observation.state", {})
    if isinstance(state_feature, dict) and "shape" in state_feature:
        state_dim = state_feature["shape"][0]
    else:
        # Fallback: try to access directly if it's already a shape tuple
        state_dim = state_feature[0] if isinstance(state_feature, (tuple, list)) else 6
        logger.warning(
            f"Could not extract state_dim from features, using default: {state_dim}"
        )

    # Get camera names from dataset features
    camera_names = [
        k.replace("observation.images.", "")
        for k in ds_features.keys()
        if k.startswith("observation.images.")
    ]

    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Camera names: {camera_names}")

    # Get image shape from first camera
    if camera_names:
        first_cam_key = f"observation.images.{camera_names[0]}"
        cam_feature = ds_features.get(first_cam_key, {})
        if isinstance(cam_feature, dict) and "shape" in cam_feature:
            image_shape = cam_feature["shape"]
        elif isinstance(cam_feature, (tuple, list)):
            image_shape = cam_feature
        else:
            image_shape = (480, 640, 3)
            logger.warning(
                f"Could not extract image shape from {first_cam_key}, using default"
            )
        logger.info(f"Image shape: {image_shape}")
    else:
        image_shape = (480, 640, 3)
        logger.warning("No cameras found in dataset, using default image shape")

    # Create observation queue
    obs_queue = ObservationQueue(obs_state_delta_frames)

    # Get robot type from policy config if available
    robot_type = getattr(policy.config, "robot_type", None)

    # Run warmup iterations
    logger.info(f"Running {warmup} warmup iterations...")
    for i in range(warmup):
        obs_frame = create_synthetic_observation(
            state_dim=state_dim,
            camera_names=camera_names,
            image_shape=image_shape,
        )
        run_inference_chunk(
            observation_frame=obs_frame,
            obs_queue=obs_queue,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            n_action_steps=n_action_steps,
            use_amp=use_amp,
            robot_type=robot_type,
        )

    # Reset queue after warmup
    obs_queue.reset()

    # Run benchmark iterations
    logger.info(f"Running {iterations} benchmark iterations...")
    stats = InferenceStats()

    for i in range(iterations):
        obs_frame = create_synthetic_observation(
            state_dim=state_dim,
            camera_names=camera_names,
            image_shape=image_shape,
        )
        _, inference_ms = run_inference_chunk(
            observation_frame=obs_frame,
            obs_queue=obs_queue,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            n_action_steps=n_action_steps,
            use_amp=use_amp,
            robot_type=robot_type,
        )
        stats.add(inference_ms)

        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i + 1}/{iterations} iterations")

    return stats


def print_results(stats: InferenceStats, device_str: str) -> None:
    """Print benchmark results to console."""
    summary = stats.summary()

    print("\n" + "=" * 80)
    print("INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Device: {device_str}")
    print("-" * 80)

    if summary["count"] == 0:
        print("No measurements recorded.")
        return

    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 80)
    print(f"{'Count':<20} {summary['count']:>15}")
    print(f"{'Mean (ms)':<20} {summary['mean']:>15.2f}")
    print(f"{'Std (ms)':<20} {summary['std']:>15.2f}")
    print(f"{'Min (ms)':<20} {summary['min']:>15.2f}")
    print(f"{'Max (ms)':<20} {summary['max']:>15.2f}")
    print(f"{'P50 (ms)':<20} {summary['p50']:>15.2f}")
    print(f"{'P95 (ms)':<20} {summary['p95']:>15.2f}")
    print(f"{'P99 (ms)':<20} {summary['p99']:>15.2f}")
    print("=" * 80)


def save_csv(stats: InferenceStats, path: Path, device_str: str) -> None:
    """Save latency measurements to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_ms", "device"])
        for i, latency_ms in enumerate(stats.latencies_ms):
            writer.writerow([i, latency_ms, device_str])

    logger.info(f"Saved latency data to {path}")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Measure inference latency for a policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arg_parser.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        required=True,
        help="Path to policy directory (pretrained_model or pretrained_model_migrated)",
    )
    arg_parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (required for policy loading)",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (None = auto-detect)",
    )
    arg_parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of inference iterations to run",
    )
    arg_parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (not counted in stats)",
    )
    arg_parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override n_action_steps from policy config",
    )
    arg_parser.add_argument(
        "--save_csv",
        type=Path,
        default=None,
        help="Path to save latency measurements as CSV",
    )

    return arg_parser.parse_args()


def main():
    args = parse_args()

    # Run benchmark
    stats = run_benchmark(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        device_str=args.device,
        iterations=args.iterations,
        warmup=args.warmup,
        n_action_steps=args.n_action_steps,
    )

    # Determine device string for output
    device_str = args.device
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    # Print results
    print_results(stats, device_str)

    # Save CSV if requested
    if args.save_csv:
        save_csv(stats, args.save_csv, device_str)


if __name__ == "__main__":
    main()
