#!/usr/bin/env python
"""Apply one exported lattice alignment to an extxyz trajectory.

The fixed transform is selected by temperature and run from the NPZ written by
``align-lattice-vectors.py``.  It rotates positions and requested Cartesian
vectors while applying the matching signed-permutation and rotation to every
cell matrix.  It never chooses a new transform for an individual frame.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import iread, write


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="input trajectory")
    parser.add_argument("-if", "--input-format", default=None, help="optional ASE input format")
    parser.add_argument("-o", "--output", type=Path, required=True, help="aligned extxyz output")
    parser.add_argument("-t", "--transforms", type=Path, required=True, help="NPZ from align-lattice-vectors.py")
    parser.add_argument("--temperature", type=float, required=True, help="temperature_K identifying the run transform")
    parser.add_argument("--run", type=int, required=True, help="run identifying the run transform")
    parser.add_argument("-f", "--frames", default=":", help="ASE frame selector (default: %(default)s)")
    parser.add_argument("--vector-array", action="append", default=[], metavar="NAME", help="per-atom Cartesian vector array to rotate; repeatable")
    parser.add_argument("--info-vector", action="append", default=[], metavar="NAME", help="three-component Cartesian vector in atoms.info to rotate; repeatable")
    return parser.parse_args()


def load_transform(path: Path, temperature: float, run: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        required = {"temperature_K", "run", "A", "R"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"transform file is missing arrays: {', '.join(sorted(missing))}")
        matches = np.flatnonzero(np.isclose(data["temperature_K"], temperature) & (data["run"] == run))
        if len(matches) != 1:
            raise ValueError(
                f"expected one transform for temperature={temperature:g} K, run={run}; found {len(matches)}"
            )
        index = matches[0]
        signed_permutation = np.asarray(data["A"][index], dtype=float)
        rotation = np.asarray(data["R"][index], dtype=float)
    if signed_permutation.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("transform matrices must have shape (3, 3)")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-8) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("transform rotation is not a proper orthogonal matrix")
    return signed_permutation, rotation


def rotate_value(value, rotation: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape[-1:] != (3,):
        raise ValueError(f"{label} must have final dimension 3; got shape {array.shape}")
    return array @ rotation


def align_atoms(atoms, signed_permutation: np.ndarray, rotation: np.ndarray, vector_arrays, info_vectors, temperature, run):
    original_cell = np.asarray(atoms.cell, dtype=float)
    atoms.positions = atoms.positions @ rotation
    atoms.set_cell(signed_permutation @ original_cell @ rotation, scale_atoms=False)
    for name in vector_arrays:
        if name not in atoms.arrays:
            raise ValueError(f"frame has no per-atom array named {name!r}")
        atoms.arrays[name][...] = rotate_value(atoms.arrays[name], rotation, f"array {name!r}")
    for name in info_vectors:
        if name not in atoms.info:
            raise ValueError(f"frame has no info vector named {name!r}")
        atoms.info[name] = rotate_value(atoms.info[name], rotation, f"info vector {name!r}")
    atoms.info["lattice_alignment_temperature_K"] = float(temperature)
    atoms.info["lattice_alignment_run"] = int(run)
    return atoms


def main() -> int:
    args = prepare_args()
    input_path, output_path, transform_path = (args.input.expanduser().resolve(), args.output.expanduser().resolve(), args.transforms.expanduser().resolve())
    try:
        if not input_path.is_file() or not transform_path.is_file():
            raise FileNotFoundError("input trajectory or transform file does not exist")
        signed_permutation, rotation = load_transform(transform_path, args.temperature, args.run)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(mode="w", prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False)
        temporary.close()
        written = 0
        try:
            read_kwargs = {"index": args.frames}
            if args.input_format is not None:
                read_kwargs["format"] = args.input_format
            for atoms in iread(input_path, **read_kwargs):
                aligned = align_atoms(atoms, signed_permutation, rotation, args.vector_array, args.info_vector, args.temperature, args.run)
                write(temporary.name, aligned, format="extxyz", append=written > 0)
                written += 1
            if written == 0:
                raise ValueError("no frames were selected")
            os.replace(temporary.name, output_path)
        except Exception:
            os.unlink(temporary.name)
            raise
        print(f"Wrote {written} aligned frame(s) to {output_path}")
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
