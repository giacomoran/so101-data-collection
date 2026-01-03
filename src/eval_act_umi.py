#!/usr/bin/env python
"""Run ACTUMI policy on SO101 robot arm.

This script evaluates an ACTUMI policy on the real SO101 robot. Unlike the standard
lerobot-record script, this is simplified for single-episode evaluation without
dataset recording.

Usage:
    python src/run_act_umi.py \
        --policy_repo_id=giacomoran/so101_data_collection_cube_hand_guided_act_umi_wrist_3 \
        --robot_port=/dev/tty.usbmodem5A460829821 \
        --camera_index=1 \
        --episode_time_s=60 \
        --fps=30

    # With device override:
    python src/run_act_umi.py \
        --policy_repo_id=giacomoran/so101_data_collection_cube_hand_guided_act_umi_wrist_3 \
        --robot_port=/dev/tty.usbmodem5A460829821 \
        --device=mps
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import torch

# Add src to path for act_umi imports
sys.path.insert(0, str(Path(__file__).parent))

from act_umi import ACTUMIPolicy

# ImageNet normalization constants (used by pretrained ResNet backbones)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


@dataclass
class RunACTUMIConfig:
    """Configuration for running ACTUMI on SO101."""

    # Policy configuration
    policy_repo_id: str = (
        "giacomoran/so101_data_collection_cube_hand_guided_act_umi_wrist_3"
    )

    # Robot configuration
    robot_port: str = "/dev/tty.usbmodem5A460829821"
    robot_id: str = "arm_follower_0"

    # Camera configuration (wrist camera)
    camera_index: int = 1
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Control configuration
    fps: int = 30
    episode_time_s: float = 60.0

    # Device (auto-detected if None)
    device: str | None = None

    # Display camera feed using rerun
    display_data: bool = False

    # Play sounds for events
    play_sounds: bool = True


def init_logging():
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_device(device: str | None) -> torch.device:
    """Auto-detect device if not specified."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_actumi_policy(repo_id: str, device: torch.device) -> ACTUMIPolicy:
    """Load ACTUMI policy from HuggingFace Hub.

    Note: The policy's relative stats are stored in Normalizer submodules
    (delta_obs_normalizer, relative_action_normalizer) and loaded automatically.
    Image normalization is handled separately using ImageNet stats.

    Returns:
        The loaded policy ready for inference.
    """
    logging.info(f"Loading ACTUMI policy from {repo_id}")

    policy = ACTUMIPolicy.from_pretrained(repo_id)
    policy.config.device = str(device)
    policy = policy.to(device)
    policy.eval()

    logging.info(f"Loaded policy with chunk_size={policy.config.chunk_size}")
    logging.info(f"  input_features: {list(policy.config.input_features.keys())}")
    logging.info(f"  n_action_steps: {policy.config.n_action_steps}")

    # Check if relative stats are loaded
    if policy.has_relative_stats:
        logging.info("  relative stats: loaded from model buffers ✓")
    else:
        logging.warning("  relative stats: NOT found in model (will use unnormalized)")

    return policy


def init_keyboard_listener():
    """Initialize keyboard listener for control events."""
    events = {
        "exit_early": False,
        "stop": False,
    }

    try:
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    logging.info("Right arrow pressed - exiting episode early")
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    logging.info("Escape pressed - stopping")
                    events["stop"] = True
                    events["exit_early"] = True
            except Exception as e:
                logging.error(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        logging.info("Keyboard listener started (Right arrow: exit, Esc: stop)")
        return listener, events

    except ImportError:
        logging.warning("pynput not available - keyboard control disabled")
        return None, events


def make_robot(cfg: RunACTUMIConfig):
    """Create and configure the SO101 robot."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config, so100_follower

    # Create camera config object
    wrist_camera = OpenCVCameraConfig(
        index_or_path=cfg.camera_index,
        fps=cfg.camera_fps,
        width=cfg.camera_width,
        height=cfg.camera_height,
    )

    # Create robot config - using so100_follower since so101 is compatible
    robot_config = so100_follower.SO100FollowerConfig(
        port=cfg.robot_port,
        id=cfg.robot_id,
        cameras={"wrist": wrist_camera},
    )

    robot = make_robot_from_config(robot_config)
    return robot


def prepare_observation(
    obs: dict,
    device: torch.device,
    motor_names: list[str],
    camera_names: list[str],
) -> dict[str, torch.Tensor]:
    """Prepare robot observation for policy inference.

    Converts raw robot observation to tensors with batch dimension.
    Images are normalized using ImageNet statistics (required for pretrained ResNet backbone).

    Args:
        obs: Raw observation from robot.get_observation()
        device: Target device for tensors
        motor_names: List of motor names in order (e.g., ["shoulder_pan.pos", ...])
        camera_names: List of camera names (e.g., ["wrist"])

    Returns:
        Dictionary with observation.state and observation.images.* as tensors
    """
    batch = {}

    # Build observation.state from individual motor positions
    state_values = [obs[motor] for motor in motor_names]
    state_array = np.array(state_values, dtype=np.float32)
    state_tensor = torch.from_numpy(state_array)
    state_tensor = state_tensor.unsqueeze(0).to(device)  # (1, state_dim)
    batch["observation.state"] = state_tensor

    # Build observation.images.* from camera images with ImageNet normalization
    # ImageNet normalization is required for pretrained ResNet backbones
    mean = IMAGENET_MEAN.view(3, 1, 1).to(device)
    std = IMAGENET_STD.view(3, 1, 1).to(device)

    for cam_name in camera_names:
        img = obs[cam_name]  # (H, W, C) uint8 RGB
        img_tensor = torch.from_numpy(img.copy()).float() / 255.0  # normalize to [0, 1]
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()  # (C, H, W)
        img_tensor = img_tensor.to(device)
        # Apply ImageNet normalization: (x - mean) / std
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        batch[f"observation.images.{cam_name}"] = img_tensor

    return batch


def run_episode(
    robot,
    policy: ACTUMIPolicy,
    events: dict,
    cfg: RunACTUMIConfig,
    device: torch.device,
):
    """Run a single episode with the policy."""
    logging.info(f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps} fps)")
    logging.info("Press Right arrow to exit early, Esc to stop completely")

    # Reset policy (clears action queue and observation history)
    policy.reset()

    # Get motor names (observation.state components) and camera names from robot
    obs_features = robot.observation_features
    motor_names = [k for k, v in obs_features.items() if v is float]
    camera_names = [k for k, v in obs_features.items() if isinstance(v, tuple)]

    logging.info(f"Motor features: {motor_names}")
    logging.info(f"Camera features: {camera_names}")

    # Get action dimension and names from robot
    action_names = list(robot.action_features.keys())
    action_dim = len(action_names)
    logging.info(f"Action space: {action_names} (dim={action_dim})")

    # Control loop
    start_time = time.perf_counter()
    step = 0
    dt_target = 1.0 / cfg.fps

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_time

        # Check termination conditions
        if events["exit_early"] or events["stop"]:
            break
        if elapsed >= cfg.episode_time_s:
            logging.info("Episode time limit reached")
            break

        # Get observation from robot
        obs = robot.get_observation()

        # Prepare observation for policy (maps raw obs to observation.state, observation.images.*)
        # This includes ImageNet normalization for images
        batch = prepare_observation(obs, device, motor_names, camera_names)

        # Run policy inference
        with torch.inference_mode():
            action = policy.select_action(batch)  # (1, action_dim)

        # Convert action to robot format
        action_np = action.squeeze(0).cpu().numpy()
        action_dict = {name: float(action_np[i]) for i, name in enumerate(action_names)}

        # Send action to robot
        robot.send_action(action_dict)

        # Logging (every 30 steps = ~1 second at 30fps)
        if step % 30 == 0:
            logging.info(
                f"Step {step}, t={elapsed:.1f}s, action[0:3]={action_np[:3].round(3)}"
            )

        step += 1

        # Sleep to maintain target fps
        loop_duration = time.perf_counter() - loop_start
        sleep_time = dt_target - loop_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

    total_time = time.perf_counter() - start_time
    actual_fps = step / total_time if total_time > 0 else 0
    logging.info(
        f"Episode complete: {step} steps in {total_time:.1f}s ({actual_fps:.1f} fps)"
    )


def log_say(message: str, play_sounds: bool, blocking: bool = False):
    """Log message and optionally speak it."""
    logging.info(message)
    if play_sounds:
        try:
            from lerobot.utils.utils import say

            say(message, blocking=blocking)
        except Exception:
            pass  # Ignore TTS errors


@draccus.wrap()
def main(cfg: RunACTUMIConfig):
    """Main entry point."""
    init_logging()
    logging.info(pformat(cfg))

    # Setup device
    device = get_device(cfg.device)
    logging.info(f"Using device: {device}")

    # Load policy
    policy = load_actumi_policy(cfg.policy_repo_id, device)

    # Create robot
    logging.info("Initializing robot...")
    robot = make_robot(cfg)

    # Setup keyboard listener
    listener, events = init_keyboard_listener()

    # Optional: initialize rerun for visualization
    if cfg.display_data:
        try:
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(session_name="run_act_umi")
        except ImportError:
            logging.warning("Rerun not available for visualization")
            cfg.display_data = False

    try:
        # Connect robot
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        # Run episode
        log_say("Starting policy evaluation", cfg.play_sounds)
        run_episode(robot, policy, events, cfg, device)

        log_say("Episode finished", cfg.play_sounds, blocking=True)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Cleanup
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()

        if listener is not None:
            listener.stop()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
