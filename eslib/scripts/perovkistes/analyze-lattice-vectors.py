#!/usr/bin/env python
"""Extract cell matrices and lattice parameters from an NpT campaign to CSV.

The campaign directory must contain folders named ``T=<temperature>K-run=<run>``.
Each CSV row represents one trajectory frame and preserves the native cell-vector
labels; alignment is intentionally deferred to ``align-lattice-vectors.py``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import iread

from lattice_vector_tools import cellpar_from_cell


RUN_RE = re.compile(r"^T=(?P<temperature>[-+]?(?:\d+(?:\.\d*)?|\.\d+))K-run=(?P<run>\d+)$")
PARAMETERS = ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg")
VECTORS = ("a_x", "a_y", "a_z", "b_x", "b_y", "b_z", "c_x", "c_y", "c_z")
HEADER = ("temperature_K", "run", "row", *PARAMETERS, *VECTORS)


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="campaign directory containing T=*K-run=* folders")
    parser.add_argument("-f", "--filename", default="i-pi.positions_0.extxyz", help="trajectory name in each run folder")
    parser.add_argument("-if", "--input-format", default="extxyz", help="ASE input format (default: %(default)s)")
    parser.add_argument("-o", "--output", type=Path, default=Path("all_lattice_vectors.csv"), help="combined CSV output")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser.parse_args()


def discover(root: Path, filename: str):
    runs = []
    for directory in root.iterdir():
        if not directory.is_dir() or (match := RUN_RE.fullmatch(directory.name)) is None:
            continue
        trajectory = directory / filename
        if trajectory.is_file():
            runs.append((float(match["temperature"]), int(match["run"]), trajectory))
        else:
            print(f"Warning: missing trajectory; skipping {trajectory}", file=sys.stderr)
    if not runs:
        raise FileNotFoundError(f"no matching runs with {filename!r} found under {root}")
    return sorted(runs)


def main() -> int:
    args = prepare_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        if not root.is_dir():
            raise FileNotFoundError(f"campaign directory not found: {root}")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
        runs = discover(root, args.filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_count = 0
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            for temperature, run, trajectory in runs:
                for row, atoms in enumerate(iread(trajectory, format=args.input_format)):
                    cell = np.asarray(atoms.cell, dtype=float)
                    cellpar = cellpar_from_cell(cell)
                    writer.writerow((temperature, run, row, *cellpar, *cell.ravel()))
                    frame_count += 1
        print(f"Wrote {frame_count} frames from {len(runs)} runs to {output}")
    except (OSError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
