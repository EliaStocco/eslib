#!/usr/bin/env python3

import argparse
import io
import sys

from ase.io import Trajectory, write


def convert_trajectory(input_file, output_file):
    """Convert every readable frame in an ASE trajectory to extxyz."""
    converted = 0

    with Trajectory(input_file, mode="r") as trajectory:
        total = len(trajectory)

        # Serialize each frame in memory first.  If either reading or extxyz
        # serialization fails, no partial frame is left in the output file.
        with open(output_file, mode="w", encoding="utf-8") as output:
            for index in range(total):
                try:
                    frame = trajectory[index]
                    buffer = io.StringIO()
                    write(buffer, frame, format="extxyz")
                except Exception as error:
                    print(
                        f"Skipping frame {index}: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
                    continue

                output.write(buffer.getvalue())
                converted += 1

    return converted, total


def main():
    parser = argparse.ArgumentParser(
        description="Convert an ASE trajectory file to extended XYZ format."
    )
    parser.add_argument("input", help="Input ASE trajectory, e.g. simulation.traj")
    parser.add_argument("output", help="Output extended XYZ file, e.g. simulation.extxyz")
    args = parser.parse_args()

    converted, total = convert_trajectory(args.input, args.output)

    if converted == 0:
        raise RuntimeError(f"No readable frames found in {args.input}")

    skipped = total - converted
    print(
        f"Converted {converted}/{total} frame(s) "
        f"({skipped} skipped): {args.input} -> {args.output}"
    )


if __name__ == "__main__":
    main()
