#!/usr/bin/env python
import argparse

import numpy as np
from ase import Atoms
from ase.io import read, write

from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Orthorhombic to hexagonal conversion of an atomic structure."

# Transformation used by hex2ortho.py.  ASE stores lattice vectors as rows.
HEX_TO_ORTHO = np.array([[1, 0, 0],
                         [1, 2, 0],
                         [0, 0, 1]])


#---------------------------------------#
def prepare_parser(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, type=str, help="input file")
    parser.add_argument("-if", "--input_format", **argv, type=str,
                        help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o", "--output", **argv, type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str,
                        help="output file format (default: %(default)s)", default=None)
    parser.add_argument("-t", "--tolerance", **argv, type=float,
                        help="distance tolerance in angstrom used to match periodic images "
                             "(default: %(default)s)", default=1e-5)
    return parser


#---------------------------------------#
def orthorhombic_to_hexagonal(atoms: Atoms, tolerance: float = 1e-5) -> Atoms:
    """Undo the supercell transformation performed by ``hex2ortho.py``.

    The forward transformation has determinant two.  Its inverse therefore
    restores the original cell and collapses each pair of periodic images to
    one atom.
    """
    if tolerance <= 0:
        raise ValueError("The matching tolerance must be positive.")
    if len(atoms) % 2 != 0:
        raise ValueError(
            "The orthorhombic structure must contain an even number of atoms."
        )

    hexagonal_cell = np.linalg.inv(HEX_TO_ORTHO) @ atoms.cell.array
    if abs(np.linalg.det(hexagonal_cell)) < 1e-12:
        raise ValueError("The input structure has a singular cell.")

    # Express every position in the restored cell.  The forward conversion
    # creates replicas only in the first two lattice directions.
    scaled = np.linalg.solve(hexagonal_cell.T, atoms.positions.T).T
    scaled[:, :2] %= 1.0

    keep = []
    unused = np.ones(len(atoms), dtype=bool)
    numbers = atoms.get_atomic_numbers()

    for index in range(len(atoms)):
        if not unused[index]:
            continue

        delta = scaled - scaled[index]
        delta[:, :2] -= np.rint(delta[:, :2])
        distances = np.linalg.norm(delta @ hexagonal_cell, axis=1)
        matches = np.flatnonzero(
            unused & (numbers == numbers[index]) & (distances <= tolerance)
        )

        if len(matches) != 2:
            raise ValueError(
                "The structure is not a hex2ortho.py supercell: expected two "
                f"periodic images of atom {index}, found {len(matches)}. "
                "Try adjusting --tolerance if the coordinates were rounded."
            )

        keep.append(index)
        unused[matches] = False

    atoms_hexagonal = atoms[keep]
    atoms_hexagonal.set_cell(hexagonal_cell, scale_atoms=False)
    atoms_hexagonal.set_scaled_positions(scaled[keep])
    atoms_hexagonal.wrap()
    return atoms_hexagonal


#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #------------------#
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms: Atoms = read(filename=args.input, format=args.input_format)
    print("done")

    #------------------#
    print("\tReducing orthorhombic cell to hexagonal cell ... ", end="")
    atoms_hexagonal = orthorhombic_to_hexagonal(atoms, args.tolerance)
    print("done")

    #------------------#
    print("\tWriting atomic structure to output file '{:s}' ... ".format(args.output), end="")
    write(filename=args.output, images=atoms_hexagonal, format=args.output_format)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()
