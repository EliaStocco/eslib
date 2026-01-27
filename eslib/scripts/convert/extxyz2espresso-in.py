#!/usr/bin/env python
import numpy as np 
from ase import Atoms
from eslib.formatting import esfmt, float_format
from eslib.classes.atomic_structures import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Print the positions and cell for a Quantum Espresso calculation given a extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"            , **argv,required=True , type=str     , help="input file")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.input,format="extxyz",index=0)[0]
    print("done")
    
    file = "positions.txt"
    print(f"\tSaving positions to {file} ... ", end="")
    with open(file,"w") as f:
        f.write("ATOMIC_POSITIONS (angstrom)\n")
        for atom in structure:
            f.write("   {:2s} {:16.8f} {:16.8f} {:16.8f}\n".format(atom.symbol,atom.x,atom.y,atom.z))
    print("done")
    
    file = "cell.txt"
    print(f"\tSaving cell to {file} ... ", end="")
    with open(file,"w") as f:
        f.write("CELL_PARAMETERS (angstrom)\n")
        for vector in structure.get_cell():
            f.write("   {:16.8f} {:16.8f} {:16.8f}\n".format(vector[0],vector[1],vector[2]))
    print("done")
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()