"""
Record a dataset using direct manipulation.

Minimally edited version of lerobot-record to support a leader-arm-only
setup.  Instead of a follower robot + leader teleoperator, a single
SO101Leader arm (torque disabled, free to move by hand) provides joint
positions that are written as BOTH observation.state AND action.

The cost of duplicating proprioception is negligible (6 floats per frame)
and it keeps the dataset fully compatible with standard LeRobot training
pipelines — no policy modifications needed.

Cameras are managed independently since there is no follower robot to
attach them to.

Usage:
    python -m so101_direct_manipulation.record.record \
        --leader.port=/dev/tty.usbmodem575E0080981 \
        --leader.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=giacomoran/cube_dm \
        --dataset.single_task="Pick up the cube and place it on the target" \
        --dataset.num_episodes=50
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DirectManipulationRecordConfig:
    leader: SO101LeaderConfig
    dataset: DatasetRecordConfig
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False


# -- Recording loop -----------------------------------------------------------


@safe_stop_image_writer
def record_loop(
    leader: SO101Leader,
    events: dict,
    fps: int,
    control_time_s: float,
    single_task: str,
    dataset: LeRobotDataset | None = None,
    display_data: bool = False,
):
    """Record one episode.

    Reads joint positions from the leader arm and writes them as both
    observation.state and action. Camera images are read from cameras
    attached to the leader config.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Read joint positions from leader arm
        raw_positions = leader.get_action()  # {motor.pos: float}

        # Read cameras
        images = {}
        for cam_name, cam in leader.cameras.items():
            images[cam_name] = cam.read_latest()

        # In direct manipulation, observation and action come from the same source
        obs_values = raw_positions | images
        observation_frame = build_dataset_frame(dataset.features, obs_values, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, raw_positions, prefix=ACTION)
        frame = {**observation_frame, **action_frame, "task": single_task}
        dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=raw_positions | images, action=raw_positions)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = 1 / fps - dt_s
        if sleep_time_s < 0:
            logging.warning(f"Record loop running at {1 / dt_s:.1f} Hz, below target {fps} Hz.")
        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


# -- Main ---------------------------------------------------------------------


@parser.wrap()
def record(cfg: DirectManipulationRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    ds = cfg.dataset
    fps = ds.fps
    play_sounds = cfg.play_sounds

    if cfg.display_data:
        init_rerun(session_name="recording")

    # --- Hardware setup ---
    leader = SO101Leader(cfg.leader)

    camera_configs = cfg.leader.cameras if cfg.leader.cameras else {}
    if not camera_configs:
        raise ValueError("At least one camera is required in --leader.cameras")

    # --- Dataset features ---
    # Mirror what lerobot-record does: run features through the processor pipeline.
    # The leader's action_features (motor positions) serve as both action and
    # observation, since in direct manipulation they are identical.
    motor_features = leader.action_features
    camera_features = {name: (cfg.height, cfg.width, 3) for name, cfg in camera_configs.items()}
    observation_features = {**motor_features, **camera_features}

    teleop_action_processor, _, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=motor_features),
            use_videos=ds.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=ds.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                ds.repo_id,
                root=ds.root,
                batch_encoding_size=ds.video_encoding_batch_size,
                vcodec=ds.vcodec,
                streaming_encoding=ds.streaming_encoding,
                encoder_queue_maxsize=ds.encoder_queue_maxsize,
                encoder_threads=ds.encoder_threads,
            )

            if len(camera_configs) > 0:
                dataset.start_image_writer(
                    num_processes=ds.num_image_writer_processes,
                    num_threads=ds.num_image_writer_threads_per_camera * len(camera_configs),
                )
        else:
            dataset = LeRobotDataset.create(
                ds.repo_id,
                fps,
                root=ds.root,
                robot_type="so101_leader",
                features=dataset_features,
                use_videos=ds.video,
                image_writer_processes=ds.num_image_writer_processes,
                image_writer_threads=ds.num_image_writer_threads_per_camera * len(camera_configs),
                batch_encoding_size=ds.video_encoding_batch_size,
                vcodec=ds.vcodec,
                streaming_encoding=ds.streaming_encoding,
                encoder_queue_maxsize=ds.encoder_queue_maxsize,
                encoder_threads=ds.encoder_threads,
            )

        # --- Connect hardware ---
        leader.connect()

        listener, events = init_keyboard_listener()

        if not ds.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it "
                "for way faster episode saving. --dataset.streaming_encoding=true "
                "--dataset.encoder_threads=2 # --dataset.vcodec=auto. More info in the documentation: "
                "https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < ds.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                record_loop(
                    leader,
                    events=events,
                    fps=fps,
                    control_time_s=ds.episode_time_s,
                    single_task=ds.single_task,
                    dataset=dataset,
                    display_data=cfg.display_data,
                )

                # Reset time between episodes (except after last)
                if not events["stop_recording"] and (
                    (recorded_episodes < ds.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", play_sounds)
                    reset_start = time.perf_counter()
                    while time.perf_counter() - reset_start < ds.reset_time_s:
                        if events["exit_early"]:
                            events["exit_early"] = False
                            break
                        precise_sleep(0.1)

                if events["rerecord_episode"]:
                    log_say("Re-record episode", play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if leader.is_connected:
            leader.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if ds.push_to_hub and dataset:
            dataset.push_to_hub(tags=ds.tags, private=ds.private)

        log_say("Exiting", play_sounds)

    return dataset


if __name__ == "__main__":
    record()
