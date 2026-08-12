"""Quantum ESPRESSO input-file writing helpers."""

from typing import Iterable

import numpy as np


def write_espresso_in(filename: str, atoms_list: Iterable, unit: str = "angstrom") -> None:
    """Write one ASE structure as a minimal Quantum ESPRESSO input fragment.

    The fragment contains the cell and atomic positions.  Positions are
    written as fractional coordinates because this representation is
    independent of the length unit used for the cell.
    """
    atoms_list = list(atoms_list)
    if len(atoms_list) != 1:
        raise ValueError("Quantum ESPRESSO input writing supports exactly one structure")

    atoms = atoms_list[0]
    unit = "angstrom" if unit is None else str(unit)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3) or np.isclose(np.linalg.det(cell), 0.0):
        raise ValueError("Quantum ESPRESSO input writing requires a non-zero cell")

    with open(filename, "w") as file:
        file.write("CELL_PARAMETERS {:s}\n".format(unit))
        for vector in cell:
            file.write("  {:.12f}  {:.12f}  {:.12f}\n".format(*vector))

        file.write("\nATOMIC_POSITIONS crystal\n")
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=False)):
            file.write("{:<2s}  {:.12f}  {:.12f}  {:.12f}\n".format(symbol, *position))
