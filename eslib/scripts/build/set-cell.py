#!/usr/bin/env python
import numpy as np
from ase.io import write
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist

from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Set the lattice vectors."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-abc", "--lengths"      , **argv, required=True , type=flist, help="list with the lattice vectors length [angstrom]")
    parser.add_argument("-ABC", "--angles"       , **argv, required=False, type=flist, help="list with the angles between the lattice vectors [deg] (default: %(default)s)" , default=[90,90,90])
    parser.add_argument("-o"  , "--output"       , **argv, required=True , type=str  , help="output file with the atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # cell
    print("\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #------------------#
    # cellpar    
    print("\tSetting the cell ... ", end="")
    for atoms in structures:
        current_cellpar = atoms.cell.cellpar()

        # Replace only provided lengths
        new_lengths = [
            args.lengths[i] if args.lengths[i] is not None else current_cellpar[i]
            for i in range(3)
        ]
        # Replace only provided angles
        new_angles = [
            args.angles[i] if args.angles[i] is not None else current_cellpar[i + 3]
            for i in range(3)
        ]

        # Create and assign new cell
        cellpar = np.array(new_lengths + new_angles, dtype=float)
        cell = Cell.fromcellpar(cellpar)

        atoms.set_pbc(True)
        atoms.set_cell(cell)
    print("done")

    
    #------------------#
    print("\n\tSaving the atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
    

#---------------------------------------#
if __name__ == "__main__":
    main()