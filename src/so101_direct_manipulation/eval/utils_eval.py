"""Shared utilities for eval scripts: base state, background observation threads, snapshot types."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from threading import Event, Lock

import numpy as np
import torch
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class SnapshotMotors:
    """Latest motor observation snapshot (immutable once created)."""

    ts: float  # perf_counter() - ts_episode_start (seconds)
    positions: dict  # {motor_name.pos: normalized_position}


@dataclass
class SnapshotCameras:
    """Latest camera observation snapshot (immutable once created)."""

    ts: float  # perf_counter() - ts_episode_start (seconds)
    images: dict  # {cam_name: HWC image array}


@dataclass
class StateBase:
    """Base shared state for all eval scripts.

    LOCKING INVARIANT
    -----------------
    All threads MUST acquire `state.lock` before:
      1. Reading or writing any field of this class (or subclasses).
      2. Performing motor bus I/O: bus.sync_read() or robot.send_action().

    Reason for (2): bus.sync_read() (motor observer) and robot.send_action() (actor)
    both access the same serial/USB motor bus. Interleaving their packets corrupts the
    bus protocol. Motor bus I/O takes ~1ms, so lock hold time is short and contention
    is low.

    CAMERA I/O EXCEPTION
    --------------------
    cam.async_read() MUST NOT be called while holding state.lock. Two reasons:
      - The camera uses a separate USB device with no shared bus with the motors,
        so there is no race condition to protect against.
      - cam.async_read() blocks for ~33ms waiting for the next frame at 30fps.
        Holding the lock for 33ms would starve the motor observer thread.

    SUBCLASSING
    -----------
    Scripts that need additional shared fields extend StateBase:

        @dataclass
        class State(StateBase):
            my_field: int = 0

            def __post_init__(self):
                super().__post_init__()  # creates lock and event_shutdown
                ...

    If the subclass defines no __post_init__, StateBase's is inherited automatically.
    """

    snapshot_motors: SnapshotMotors | None = None
    snapshot_cameras: SnapshotCameras | None = None
    event_shutdown: Event | None = None
    lock: Lock | None = None

    def __post_init__(self):
        self.event_shutdown = Event()
        self.lock = Lock()


class ThreadObsMotors(threading.Thread):
    """Background thread that reads motor positions at fps_obs.

    Acquires state.lock for the entire duration of bus.sync_read() AND the
    subsequent snapshot write. This ensures robot.send_action() in the actor
    (also under state.lock) cannot interleave with a motor bus read.

    Lock hold time per frame: ~1ms (bus.sync_read) + ~0.1ms (snapshot write).
    """

    def __init__(
        self,
        bus,
        fps_obs: int,
        ts_episode_start: float,
        state: StateBase,
        is_logging_rr: bool,
    ):
        super().__init__(name="ObsMotors", daemon=True)
        self._bus = bus
        self._fps_obs = fps_obs
        self._ts_episode_start = ts_episode_start
        self._state = state
        self._is_logging_rr = is_logging_rr

    def run(self) -> None:
        from so101_direct_manipulation.eval.utils_rerun import log_rr_obs_motors

        duration_s_frame_target = 1.0 / self._fps_obs
        while not self._state.event_shutdown.is_set():
            ts_frame_start = time.perf_counter()
            ts = ts_frame_start - self._ts_episode_start

            # Lock covers bus I/O + snapshot write atomically.
            # See StateBase docstring for the locking invariant.
            with self._state.lock:
                positions = self._bus.sync_read("Present_Position")
                positions = {f"{motor}.pos": val for motor, val in positions.items()}
                self._state.snapshot_motors = SnapshotMotors(ts=ts, positions=positions)

            if self._is_logging_rr:
                log_rr_obs_motors(ts, positions)

            duration_s_frame = time.perf_counter() - ts_frame_start
            duration_s_sleep = duration_s_frame_target - duration_s_frame
            if duration_s_sleep > 0:
                precise_sleep(duration_s_sleep)


class ThreadObsCameras(threading.Thread):
    """Background thread that reads camera frames in a tight loop.

    cam.async_read() is called OUTSIDE state.lock because:
      - It blocks ~33ms at 30fps, which would starve the motor observer.
      - The camera is a separate USB device with no shared bus with the motors.
    Only the snapshot write (~0.1ms) acquires state.lock.

    See StateBase docstring for the full camera I/O exception rationale.
    """

    def __init__(
        self,
        cameras: dict,
        ts_episode_start: float,
        state: StateBase,
        is_logging_rr: bool,
    ):
        super().__init__(name="ObsCamera", daemon=True)
        self._cameras = cameras
        self._ts_episode_start = ts_episode_start
        self._state = state
        self._is_logging_rr = is_logging_rr

    def run(self) -> None:
        from so101_direct_manipulation.eval.utils_rerun import log_rr_obs_cameras

        idx_frame_camera = 0
        while not self._state.event_shutdown.is_set():
            ts_frame_start = time.perf_counter()
            ts = ts_frame_start - self._ts_episode_start

            # cam.async_read() blocks ~33ms — MUST NOT hold state.lock.
            # See StateBase docstring for the camera I/O exception.
            images = {}
            for name, camera in self._cameras.items():
                images[name] = camera.async_read()

            # Only the snapshot write needs the lock (~0.1ms).
            with self._state.lock:
                self._state.snapshot_cameras = SnapshotCameras(ts=ts, images=images)

            if self._is_logging_rr:
                log_rr_obs_cameras(ts, idx_frame_camera, images)

            idx_frame_camera += 1


def build_action_robot(action_policy: torch.Tensor, names_action: list[str]) -> dict[str, float]:
    """Convert an unnormalized CPU action tensor [action_dim] to a robot action dict."""
    return {name: float(action_policy[i]) for i, name in enumerate(names_action)}


def build_obs_policy(
    positions: dict,
    images: dict,
    names_motor: list[str],
    names_camera: list[str],
) -> dict:
    """Assemble observation dict for policy inference from motor positions and camera images.

    This function does not acquire any lock.
    """
    obs_policy = {}
    obs_policy["observation.state"] = np.array([positions[name] for name in names_motor], dtype=np.float32)
    for name in names_camera:
        obs_policy[f"observation.images.{name}"] = images[name]
    return obs_policy
