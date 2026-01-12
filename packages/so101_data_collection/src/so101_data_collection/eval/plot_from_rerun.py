#!/usr/bin/env python3
"""
Plot observations and actions from an exported ReRun recording.

Usage:
    python plot_from_rerun.py <path_to_rrd_file> [--output OUTPUT_PATH]

The script automatically extracts motor names from the recording.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import rerun as rr
from so101_data_collection.eval.rerun_utils import plot_observation_actions_from_rerun


def main():
    parser = argparse.ArgumentParser(
        description="Plot observations and actions from a ReRun recording"
    )
    parser.add_argument("rrd_path", type=Path, help="Path to the .rrd recording file")
    parser.add_argument("--output", "-o", type=Path, help="Output path for the plot")
    args = parser.parse_args()

    if not args.rrd_path.exists():
        print(f"Error: File {args.rrd_path} does not exist")
        sys.exit(1)

    print(f"Loading recording: {args.rrd_path}")

    recording = rr.dataframe.load_recording(str(args.rrd_path))
    schema = recording.schema()
    component_cols = schema.component_columns()

    motor_names = []
    for col in component_cols:
        if col.entity_path.startswith('/observation') and 'Scalars:scalars' in col.component:
            motor_name = col.entity_path.split('/')[-1].split('.')[0]
            motor_names.append(f"{motor_name}.pos")

    motor_names.sort()
    print(f"Found {len(motor_names)} motors: {motor_names}")

    output_path = args.output if args.output else args.rrd_path.with_suffix('.png')

    try:
        plot_observation_actions_from_rerun(
            recording_path=args.rrd_path,
            motor_names=motor_names,
            output_path=output_path,
        )
        print("✓ Plot generated successfully")
        print(f"  Saved to: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
