"""ReRun visualization utilities with slash-separated paths for blueprint matching.

This module provides custom logging functions that use "/" instead of "." in entity paths
to match blueprint path patterns. This ensures proper matching in ReRun blueprints.
"""

import numbers

import numpy as np
import rerun as rr


def _is_scalar(x):
    """Check if value is a scalar."""
    return isinstance(x, (float, numbers.Real, np.integer, np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, np.ndarray] | None = None,
    action: dict[str, np.ndarray] | None = None,
) -> None:
    """Log observation and action data to ReRun using slash-separated paths.

    This is a custom version of lerobot's log_rerun_data that uses "/" instead of "." in paths
    to match blueprint path patterns. This makes blueprint matching work correctly.

    Paths logged:
    - Observations: /observation/state_0, /observation/state_1, ..., /observation/images/wrist, etc.
    - Actions: /action/shoulder_pan/pos, /action/shoulder_lift/pos, etc.

    Args:
        observation: Optional dictionary containing observation data to log.
        action: Optional dictionary containing action data to log.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue

            # Convert key to use slashes instead of dots
            # observation.state -> /observation/state
            # observation.images.wrist -> /observation/images/wrist
            if k.startswith("observation."):
                key = "/" + k.replace(".", "/")
            elif k.startswith("observation"):
                key = "/observation/" + k.replace("observation", "").lstrip(".")
            else:
                key = "/observation/" + k

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if (
                    arr.ndim == 3
                    and arr.shape[0] in (1, 3, 4)
                    and arr.shape[-1] not in (1, 3, 4)
                ):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    # Log each element: /observation/state_0, /observation/state_1, etc.
                    # Use underscore to match original pattern but with slashes
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue

            # Convert key to use slashes instead of dots
            # shoulder_pan.pos -> /action/shoulder_pan/pos
            if k.startswith("action."):
                key = "/" + k.replace(".", "/")
            else:
                key = "/action/" + k.replace(".", "/")

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
