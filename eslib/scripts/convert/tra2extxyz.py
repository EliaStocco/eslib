#!/usr/bin/env python3

import argparse
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(
        description="Convert an ASE trajectory file to extended XYZ format."
    )
    parser.add_argument("input", help="Input ASE trajectory, e.g. simulation.traj")
    parser.add_argument("output", help="Output extended XYZ file, e.g. simulation.extxyz")
    args = parser.parse_args()

    frames = read(args.input, index=":")
    write(args.output, frames, format="extxyz")

    print(f"Converted {len(frames)} frame(s): {args.input} -> {args.output}")


if __name__ == "__main__":
    main()