#!/usr/bin/env python
"""Put the atoms assigned to each perovskite unit cell in nearby images."""

import os
import tempfile

import numpy as np
from ase.geometry import find_mic
from ase.io import iread, write

from eslib.formatting import esfmt


# --------------------------------------- #
description = (
    "Wrap the atoms of each labelled perovskite unit cell into nearby "
    "periodic images."
)


# --------------------------------------- #
def prepare_args(description):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i", "--input", **argv, required=True, type=str,
        help="input extxyz trajectory",
    )
    parser.add_argument(
        "-if", "--input_format", **argv, required=False, type=str,
        default=None, help="input file format (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--indices", **argv, required=True, type=str,
        help=(
            "unit-cells.txt produced by divide-perovskite-into-unitcells.py; "
            "only its first (unit-cell label) column is used"
        ),
    )
    parser.add_argument(
        "-f", "--frames", **argv, required=False, type=str, default=":",
        help="frames to process (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", **argv, required=False, type=str,
        default="wrapped-perovskite-unitcells.extxyz",
        help="output extxyz trajectory (default: %(default)s)",
    )
    return parser


# --------------------------------------- #
def read_unit_cell_labels(filename):
    """Read and validate the first column of the divider output."""
    data = np.loadtxt(filename, ndmin=2)
    if data.shape[1] < 1:
        raise ValueError(f"'{filename}' does not contain a unit-cell-label column")

    labels = data[:, 0]
    if not np.all(np.isfinite(labels)) or not np.allclose(labels, np.rint(labels)):
        raise ValueError("Unit-cell labels in the first column must be integers")
    return np.rint(labels).astype(int)


# --------------------------------------- #
def wrap_unit_cells(atoms, labels):
    """Return ``atoms`` after bringing each label group close to its first atom.

    ``find_mic`` chooses the closest periodic image of every atom relative to
    the group's reference atom.  Each resulting displacement is a whole
    lattice-vector translation, and therefore leaves fractional coordinates
    unchanged modulo one.
    """
    if len(labels) != len(atoms):
        raise ValueError(
            f"The indices file contains {len(labels)} labels, but the structure "
            f"contains {len(atoms)} atoms"
        )
    if not np.any(atoms.pbc):
        raise ValueError("The input structure has no periodic directions to wrap")

    original_scaled_positions = atoms.get_scaled_positions(wrap=False)
    positions = atoms.positions.copy()
    for label in np.unique(labels):
        atom_indices = np.flatnonzero(labels == label)
        reference = positions[atom_indices[0]]
        displacement, _ = find_mic(
            positions[atom_indices] - reference, atoms.cell, pbc=atoms.pbc
        )
        positions[atom_indices] = reference + displacement

    atoms.positions = positions

    # Guard against changing a coordinate by anything other than a lattice
    # vector.  This also catches malformed (e.g. singular) simulation cells.
    translation = atoms.get_scaled_positions(wrap=False) - original_scaled_positions
    periodic = np.asarray(atoms.pbc, dtype=bool)
    if not np.allclose(translation[:, periodic], np.rint(translation[:, periodic])):
        raise RuntimeError("Wrapping changed periodic fractional coordinates")
    if not np.allclose(translation[:, ~periodic], 0.0):
        raise RuntimeError("Wrapping changed non-periodic fractional coordinates")
    return atoms


# --------------------------------------- #
@esfmt(prepare_args, description)
def main(args):
    labels = read_unit_cell_labels(args.indices)
    print(
        "\tRead {:d} atom labels for {:d} unit cells from '{}'".format(
            len(labels), len(np.unique(labels)), args.indices
        )
    )

    read_kwargs = {"index": args.frames}
    if args.input_format is not None:
        read_kwargs["format"] = args.input_format

    # A trajectory is read lazily.  Writing directly to ``args.output`` when
    # it names the input would truncate the input before all frames have been
    # consumed.  A temporary file also prevents a failed run from leaving a
    # partial output trajectory behind.
    output_directory = os.path.dirname(os.path.abspath(args.output))
    output_basename = os.path.basename(args.output)
    temporary_output = tempfile.NamedTemporaryFile(
        mode="w", prefix=f".{output_basename}.", suffix=".tmp",
        dir=output_directory, delete=False,
    )
    temporary_output.close()

    print("\tWrapping unit cells from '{}' ...".format(args.input))
    output_written = False
    n_frames = 0
    try:
        for atoms in iread(args.input, **read_kwargs):
            wrap_unit_cells(atoms, labels)
            write(temporary_output.name, atoms, format="extxyz", append=output_written)
            output_written = True
            n_frames += 1

        if n_frames == 0:
            raise ValueError("No frames were selected from the input trajectory")
        os.replace(temporary_output.name, args.output)
    except Exception:
        os.unlink(temporary_output.name)
        raise
    print("\tWrote {:d} wrapped frame(s) to '{}'".format(n_frames, args.output))


# --------------------------------------- #
if __name__ == "__main__":
    main()
