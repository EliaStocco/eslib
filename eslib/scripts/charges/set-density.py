#!/usr/bin/env python
import numpy as np
from ase.io import write
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist, str2bool
from eslib.physics import compute_density

from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Set the density of some atomic structures."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str     , help="file with the atomic structures")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str     , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-d"  , "--density"      , **argv, required=True , type=flist   , help="list with the lattice vectors length [g/cm^3]")
    parser.add_argument("-s"  , "--scale"        , **argv, required=False, type=str2bool, help="whether to scale the positions (default: %(default)s)", default=True)
    parser.add_argument("-o"  , "--output"       , **argv, required=True , type=str     , help="output file with the atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str     , help="output file format (default: %(default)s)", default=None)
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
    print("\tModifying the density of all the structures... ")
    for n,atoms in enumerate(structures):
        old_density = compute_density(atoms)
        # frac = atoms.get_scaled_positions()
        # volume = atoms.get_volume()
        factor = (args.density/old_density) ** (1. / 3)
        cellpar = atoms.cell.cellpar()
        cellpar[:3] /= factor
        cell = Cell.fromcellpar(cellpar)
        atoms.set_cell(cell,scale_atoms=args.scale)
        new_density = compute_density(atoms)
        print(f"\t - {n}) {old_density:6} --> {new_density:6}")
        assert np.allclose(new_density,args.density), "coding error"
    
    #------------------#
    print("\n\tSaving the atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
    

#---------------------------------------#
if __name__ == "__main__":
    main()