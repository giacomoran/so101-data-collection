"""ReRun visualization utilities with slash-separated paths for blueprint matching.

This module provides custom logging functions that use "/" instead of "." in entity paths
to match blueprint path patterns. This ensures proper matching in ReRun blueprints.
"""

import logging
import numbers
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr


def _is_scalar(x):
    """Check if value is a scalar."""
    return isinstance(x, (float, numbers.Real, np.integer, np.floating)) or (isinstance(x, np.ndarray) and x.ndim == 0)


def log_rerun_data(
    observation: dict[str, np.ndarray] | None = None,
    action: dict[str, np.ndarray] | None = None,
    timestep: int | None = None,
    idx_chunk: int | None = None,
) -> None:
    """Log observation and action data to ReRun using slash-separated paths.

    This is a custom version of lerobot's log_rerun_data that uses "/" instead of "." in paths
    to match blueprint path patterns. This makes blueprint matching work correctly.

    Paths logged:
    - Observations: /observation/state_0, /observation/state_1, ..., /observation/images/wrist, etc.
    - Actions: /action/shoulder_pan/pos, /action/shoulder_lift/pos, etc.
    - Timestep: /timestep (the control timestep, not display frame index)
    - Chunk index: /idx_chunk (the current chunk being executed)

    Args:
        observation: Optional dictionary containing observation data to log.
        action: Optional dictionary containing action data to log.
        timestep: Optional control timestep to log.
        idx_chunk: Optional chunk index to log.
    """
    if timestep is not None:
        rr.log("/timestep", rr.Scalars(float(timestep)))

    if idx_chunk is not None:
        rr.log("/idx_chunk", rr.Scalars(float(idx_chunk)))
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
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
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


def plot_observation_actions_from_rerun(
    recording_path: Path = None,
    motor_names: list[str] = None,
    dummy_observations: list[np.ndarray] = None,
    dummy_actions: list[np.ndarray] = None,
    observation_names: list[str] = None,
    display_frame_times_ms: np.ndarray = None,
    control_timestep_vals: np.ndarray = None,
    control_times_ms: np.ndarray = None,
    idx_chunk_indices: list[int] = None,
    output_path: Path | None = None,
) -> None:
    """Plot observations and actions from a rerun recording file.

    This function creates a plot where:
    - Observations are plotted as lines at display timesteps (higher frequency, e.g. 30fps)
    - Actions are plotted as dots at control timesteps (lower frequency, e.g. 10fps)
    - X-axis is labeled with control timestep indices
    - Vertical red lines mark chunk boundaries

    The x-axis uses display frame indices as the underlying scale, but tick labels
    show the corresponding control timestep indices.

    Args:
        recording_path: Path to the .rrd recording file.
        motor_names: List of motor names from robot.observation_features (e.g., ["shoulder_pan.pos", ...]).
        dummy_observations: For testing - list of observation arrays per joint.
        dummy_actions: For testing - list of action arrays per joint.
        observation_names: For testing - list of joint names.
        display_frame_times_ms: For testing - array of display frame timestamps in ms.
        control_timestep_vals: For testing - array of control timestep indices (0, 1, 2, ...).
        control_times_ms: For testing - array of control timestep timestamps in ms.
        idx_chunk_indices: For testing - list of control timestep indices where chunks switch.
        output_path: Optional path to save the plot. If None, shows the plot interactively.
    """
    # Handle dummy data mode
    if recording_path is None and all(
        x is not None
        for x in [
            dummy_observations,
            dummy_actions,
            observation_names,
            display_frame_times_ms,
            control_timestep_vals,
            control_times_ms,
            idx_chunk_indices,
        ]
    ):
        logging.info("Using dummy data mode")
        observations = dummy_observations
        actions = dummy_actions
        motor_names = observation_names
        idx_chunks = idx_chunk_indices

        # X-axis: display frame indices (0, 1, 2, ..., n_display_frames-1)
        n_display_frames = len(display_frame_times_ms)
        n_control_steps = len(control_timestep_vals)
        display_ratio = n_display_frames // n_control_steps  # e.g., 3 display frames per control step

        # Observations: plot at display frame indices
        obs_x_axis = np.arange(n_display_frames)

        # Actions: plot at display frame indices corresponding to control timesteps
        # Control step 0 -> display frame 0, control step 1 -> display frame 3, etc.
        action_x_axis = control_timestep_vals * display_ratio

        # Chunk switches: convert from control timestep to display frame index
        idx_chunk_x = [idx * display_ratio for idx in idx_chunks]

        # For x-axis tick labels: show control timesteps at their positions
        # Create tick positions at control timestep boundaries
        tick_positions = control_timestep_vals * display_ratio
        tick_labels = control_timestep_vals

    else:
        logging.info(f"Loading rerun recording: {recording_path}")

        # Load recording
        recording = rr.dataframe.load_recording(str(recording_path))
        logging.info("Recording loaded, creating view...")

        # Get available index columns
        schema = recording.schema()
        index_cols = [col.name for col in schema.index_columns()]

        # Determine index to use (prefer frame_nr, fall back to log_tick)
        index_name = "frame_nr" if "frame_nr" in index_cols else index_cols[0] if index_cols else "log_tick"
        logging.info(f"Using index: {index_name}")

        # Create view of observation and action data
        view = recording.view(index=index_name, contents="/**").fill_latest_at()
        logging.info("View created, querying data...")

        # Query data
        table = view.select().read_all()
        logging.info(f"Query completed, got {table.num_rows} rows of data")

        # Filter to only scalar columns before converting to pandas
        scalar_cols = [col for col in table.schema.names if ":Scalars:scalars" in col and col != index_name]
        scalar_table = table.select(scalar_cols)
        df = scalar_table.to_pandas()
        logging.info(f"Filtered to {len(scalar_cols)} scalar columns, converted to pandas")

        # Add index column to dataframe
        index_table = table.select([index_name])
        index_df = index_table.to_pandas()
        df[index_name] = index_df[index_name]

        # Extract timestep values (control timestep indices)
        timestep_col = "/timestep:Scalars:scalars"

        def _extract_scalar(x):
            if isinstance(x, list) and len(x) > 0:
                return float(x[0])
            elif isinstance(x, np.ndarray) and x.ndim == 0:
                return float(x)
            else:
                return float(x)

        if timestep_col in df.columns:
            timestep_vals = df[timestep_col].apply(_extract_scalar).values
        else:
            timestep_vals = None

        # X-axis: use display frame indices (row indices in the dataframe)
        n_display_frames = len(df)
        obs_x_axis = np.arange(n_display_frames)

        # Find unique control timesteps and their first occurrence (display frame index)
        if timestep_vals is not None:
            # Get first display frame index for each unique control timestep
            unique_timestep, first_indices = np.unique(timestep_vals, return_index=True)
            action_x_axis = first_indices  # Display frame indices where each control step starts

            # For x-axis ticks: show control timesteps
            # Sample every N control steps to avoid crowding
            n_ticks = min(20, len(unique_timestep))
            tick_step = max(1, len(unique_timestep) // n_ticks)
            tick_positions = first_indices[::tick_step]
            tick_labels = unique_timestep[::tick_step].astype(int)
        else:
            # Fallback: no t values, use frame indices directly
            action_x_axis = obs_x_axis
            tick_positions = np.linspace(0, n_display_frames - 1, 10).astype(int)
            tick_labels = tick_positions

        # Extract chunk switch positions
        idx_chunk_col = "/idx_chunk:Scalars:scalars"
        idx_chunk_x = []
        if idx_chunk_col in df.columns:
            chunk_data = df[idx_chunk_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
            chunk_data = chunk_data.ffill().fillna(-1)
            for i in range(1, len(chunk_data)):
                if chunk_data.iloc[i] != chunk_data.iloc[i - 1]:
                    # Record the display frame index where chunk changed
                    idx_chunk_x.append(i)

    # Setup plot
    fig, ax1 = plt.subplots(figsize=(16, 10))
    n_joints = len(motor_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    logging.info("Plot setup complete, starting data plotting...")

    for joint_idx, motor_key in enumerate(motor_names):
        color = colors[joint_idx]
        base_name = motor_key.split(".")[0] if "." in motor_key else motor_key

        if recording_path is not None:
            # Real data processing
            obs_col = f"/observation/{base_name}.pos:Scalars:scalars"
            if obs_col in df.columns:
                obs_data = df[obs_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                # Plot observations at display frame indices (continuous line)
                ax1.plot(obs_x_axis, obs_data.values, color=color, linewidth=0.8, linestyle="-", alpha=0.7)

            action_col = f"/action/{base_name}/pos:Scalars:scalars"
            if action_col in df.columns:
                action_data = df[action_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

                if timestep_vals is not None:
                    # Get action value at each unique control timestep
                    # Use the first occurrence of each t value
                    action_vals_at_t = action_data.values[action_x_axis]
                    mask = action_data.notna().values[action_x_axis]
                    ax1.scatter(action_x_axis[mask], action_vals_at_t[mask], color=color, s=15, label=base_name)
                else:
                    mask = action_data.notna()
                    ax1.scatter(obs_x_axis[mask], action_data[mask], color=color, s=15, label=base_name)
        else:
            # Dummy data processing
            if joint_idx < len(observations):
                # Observations: plot at display frame indices (line)
                ax1.plot(obs_x_axis, observations[joint_idx], color=color, linewidth=0.8, linestyle="-", alpha=0.7)

            if joint_idx < len(actions):
                # Actions: plot at control timestep positions (dots)
                mask = ~np.isnan(actions[joint_idx])
                ax1.scatter(action_x_axis[mask], actions[joint_idx][mask], color=color, s=15, label=base_name)

    # Draw vertical lines at chunk boundaries
    for x in idx_chunk_x:
        ax1.axvline(x=x, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    # Set x-axis ticks to show control timesteps
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="idx_chunk switch")
    ]
    for joint_idx, motor_key in enumerate(motor_names):
        color = colors[joint_idx]
        base_name = motor_key.split(".")[0] if "." in motor_key else motor_key
        legend_elements.append(Line2D([0], [0], color=color, linestyle="", marker="o", markersize=6, label=base_name))

    ax1.legend(handles=legend_elements, loc="best", fontsize=10, ncol=2)

    ax1.set_xlabel("Control timestep")
    ax1.set_ylabel("Position (rad)")
    ax1.set_title("Observations (lines at display fps) and Actions (dots at control fps)")
    ax1.grid(True, alpha=0.3)

    # Set x-axis limits
    if recording_path is not None:
        ax1.set_xlim(0, n_display_frames - 1)
    else:
        ax1.set_xlim(0, len(obs_x_axis) - 1)

    plt.tight_layout()

    if output_path:
        logging.info(f"Saving plot to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Plot saved to {output_path}")
    else:
        logging.warning("No output path provided, skipping plot save")
