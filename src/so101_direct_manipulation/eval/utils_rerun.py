"""ReRun visualization utilities with slash-separated paths for blueprint matching.

This module provides custom logging functions that use "/" instead of "." in entity paths
to match blueprint path patterns.

## Recording Schema

Reference for the columns logged in `.rrd` files produced by the eval scripts.

The only rerun timeline is `timestamp` (set via `rr.set_time("timestamp", duration=ts)`).
All threads log independently: the motor observer at `fps_obs`, the camera observer at
camera fps, and the actor at the action rate. These streams are interleaved in the
`timestamp` timeline.

### Timing Columns

- `log_time`: Rerun wall-clock at the moment `rr.log()` is called. **Do not use for
  analysis** — see log_time jitter note below.
- `timestamp`: `perf_counter` seconds relative to episode start. Use this as the time
  axis for analysis.

### Observation Columns (logged by motor/camera observer threads)

- `/observation/{motor}.pos`: Motor joint position in degrees (e.g.
  `/observation/shoulder_pan.pos`). Logged continuously by `ThreadObsMotors` at `fps_obs`.
- `/observation/images/{camera}`: Camera image, logged `static=True`. Not in the timeline
  index.
- `/idx_frame_camera`: Camera frame counter. Logged alongside each camera image by
  `ThreadObsCameras`.

### Action Columns (logged by actor thread)

- `/action/{motor}/pos`: Commanded joint position (e.g. `/action/shoulder_pan/pos`).
  Logged only when an action is sent: every policy step in eval_sync/eval_sync_discard;
  control + interpolation frames in eval_async_smooth.

### Control Columns (logged by actor thread alongside actions)

- `/timestep`: Policy step counter. Logged alongside each action, **not** at motor
  observation frames.
- `/idx_chunk`: Active chunk index. Logged alongside each action; increments each time a
  new inference chunk is loaded.

### Data Stream Summary

| Stream           | Rate         | Columns logged |
|------------------|--------------|----------------|
| Motor observer   | `fps_obs`    | `/observation/{motor}.pos` |
| Camera observer  | camera fps   | `/observation/images/{camera}`, `/idx_frame_camera` |
| Actor (eval_sync) | `fps_policy` | `/timestep`, `/idx_chunk`, `/action/{motor}/pos` |
| Actor (eval_async_smooth) | `fps_policy` (control) + `fps_interpolation` (interp) | same |

### log_time Jitter

`log_time` has non-uniform spacing because the motor observer and actor are separate
threads that call `rr.log()` at different moments within the same wall-clock interval.
This creates (tiny_gap, large_gap) pairs in `log_time` that sum to one control period —
roughly 5% of consecutive frame pairs appear only 1–5 ms apart instead of the expected
~18 ms at 55 Hz.

The `log_time` delta distribution is therefore bimodal and cannot be used as a uniform
time axis. Use `timestamp` instead.

### How to Use for Analysis

- **Time axis**: use `timestamp` (uniform spacing within each stream).
- **Commanded frames**: rows where `/action/{motor}/pos` is non-null.
- **Policy-output frames**: first occurrence of each unique `/timestep` value among action
  rows (one per new inference chunk step).
- **Identify active chunk**: use `/idx_chunk` (present only at action frames; ffill to
  propagate to other rows).
"""

import numpy as np
import rerun as rr


def log_rr_obs_motors(ts: float, value: dict) -> None:
    """Called by ThreadObsMotors. Logs motor positions to rerun.

    Args:
        timestamp: Elapsed time since episode start (perf_counter relative, seconds).
        value: Dict of {motor_name.pos: value} motor positions.
    """
    rr.set_time("timestamp", duration=ts)
    for motor_key, val in value.items():
        rr.log(f"/observation/{motor_key}", rr.Scalars(float(val)))


def log_rr_obs_cameras(ts: float, idx_frame_camera: int, value: dict) -> None:
    """Called by ThreadObsCameras. Logs camera images and frame counter to rerun.

    Args:
        ts: Elapsed time since episode start (perf_counter relative, seconds).
        idx_frame_camera: Camera frame counter (used for timing diagnostics).
        value: Dict of {cam_name: value} camera images.
    """
    rr.set_time("timestamp", duration=ts)
    rr.log("/idx_frame_camera", rr.Scalars(float(idx_frame_camera)))
    for cam_name, frame in value.items():
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        rr.log(f"/observation/images/{cam_name}", rr.Image(frame), static=True)


def log_rr_action(
    ts: float,
    timestep: int,
    idx_chunk: int,
    value: dict,
) -> None:
    """Called by Actor. Logs action data to rerun.

    A timestep is one policy step: one observation captured and one action sent.
    Matches the training dataset's notion of timestep (e.g. 30 timesteps/s).

    Paths logged: /timestep, /idx_chunk, /action/{motor_name}

    Args:
        ts: Elapsed time since episode start (perf_counter relative, seconds).
        timestep: Current policy timestep counter (increments once per control frame).
        idx_chunk: Current chunk index (only at chunk start, else None).
        value: Dict of {action_name: value} joint positions.
    """
    rr.set_time("timestamp", duration=ts)
    rr.log("/timestep", rr.Scalars(float(timestep)))
    rr.log("/idx_chunk", rr.Scalars(float(idx_chunk)))
    for k, v in value.items():
        if k.startswith("action."):
            key = "/" + k.replace(".", "/")
        else:
            key = "/action/" + k.replace(".", "/")
        rr.log(key, rr.Scalars(float(v)))
