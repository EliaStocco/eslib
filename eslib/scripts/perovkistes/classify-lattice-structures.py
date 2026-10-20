#!/usr/bin/env python
"""Classify perovskite cells in a lattice-parameter summary CSV.

The classifier uses the metric averaged in each input row.  It is intended for
the output of ``align-lattice-vectors.py`` (or an equivalent CSV containing
``a_angstrom_mean`` ... ``gamma_deg_mean``).  The labels are the four common
BaTiO3 phases: cubic, tetragonal, orthorhombic, and rhombohedral.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


LENGTHS = ("a_angstrom_mean", "b_angstrom_mean", "c_angstrom_mean")
ANGLES = ("alpha_deg_mean", "beta_deg_mean", "gamma_deg_mean")


def classify(row: dict[str, str], length_tolerance: float, angle_tolerance: float) -> str:
    lengths = np.array([float(row[name]) for name in LENGTHS])
    angles = np.array([float(row[name]) for name in ANGLES])
    if not np.isfinite(lengths).all() or not np.isfinite(angles).all() or np.any(lengths <= 0):
        raise ValueError("non-finite or non-positive lattice parameters")

    # Compare lengths relative to the mean so the same thresholds work for
    # primitive cells and supercells.  Angles are compared in degrees.
    relative = np.abs(lengths[:, None] - lengths[None, :]) / lengths.mean()
    equal_lengths = relative <= length_tolerance
    # A flipped lattice vector changes an angle from 90+d to 90-d without
    # changing the lattice metric.  Therefore compare the magnitudes of the
    # deviations from 90 degrees, rather than the signed angles themselves.
    angle_deviations = np.abs(angles - 90.0)
    right_angles = angle_deviations <= angle_tolerance
    equal_angle_deviations = np.abs(
        angle_deviations[:, None] - angle_deviations[None, :]
    ) <= angle_tolerance

    if equal_lengths.all() and right_angles.all():
        return "cubic"
    if right_angles.all():
        # Exactly two length groups is the tetragonal signature; three groups
        # is orthorhombic.  A near-boundary cell is assigned to the nearest
        # of these two by the number of matching pairs.
        matching_pairs = int(equal_lengths[np.triu_indices(3, 1)].sum())
        return "tetragonal" if matching_pairs >= 1 else "orthorhombic"
    if (
        equal_lengths.all()
        and equal_angle_deviations.all()
        and np.all(angle_deviations > angle_tolerance)
    ):
        return "rhombohedral"

    raise ValueError(
        "cell does not match one of the four requested metrics; "
        "adjust --length-tolerance/--angle-tolerance or inspect the row"
    )


def input_files(path: Path):
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files found in {path}")
        return files
    raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="summary CSV, or directory containing summary CSVs")
    parser.add_argument("-o", "--output", type=Path, default=Path("lattice_structure.csv"))
    parser.add_argument("--parameter-column", default="temperature_K", help="column to retain as the parameter")
    parser.add_argument("--length-tolerance", type=float, default=0.005, help="relative length tolerance (default: %(default)s)")
    parser.add_argument("--angle-tolerance", type=float, default=0.25, help="angle tolerance in degrees (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        if args.length_tolerance <= 0 or args.angle_tolerance <= 0:
            raise ValueError("tolerances must be positive")
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
        rows = []
        for filename in input_files(args.input.expanduser().resolve()):
            with filename.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                required = {args.parameter_column, *LENGTHS, *ANGLES}
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"{filename}: missing columns: {', '.join(sorted(missing))}")
                for row in reader:
                    row["structure"] = classify(row, args.length_tolerance, args.angle_tolerance)
                    rows.append({args.parameter_column: row[args.parameter_column], "structure": row["structure"]})
        rows.sort(key=lambda row: float(row[args.parameter_column]))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(args.parameter_column, "structure"))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} classifications to {output}")
    except (OSError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
