#!/usr/bin/env python
import argparse
import numpy as np
from ase.build import make_supercell
from ase import Atoms
from ase.io import read, write
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Hexagonal to orthorhombic conversion of a atomic structure."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv,type=str, help="input file")
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o"  , "--output"       ,   **argv,type=str, help="output file")
    parser.add_argument("-of" , "--output_format",   **argv,type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from input file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = read(filename=args.input,format=args.input_format)
    print("done")
    
    # ----- Hexagonal → Orthorhombic transformation matrix -----
    M = np.array([[1, 0, 0],
                  [1, 2, 0],
                  [0, 0, 1]])

    print("\tBuilding orthorhombic supercell ... ", end="")
    atoms_ortho = make_supercell(atoms, M)
    print("done")

    # Optional: wrap atoms inside the cell
    atoms_ortho.wrap()

    #------------------#
    print("\tWriting atomic structures to output file '{:s}' ... ".format(args.output), end="")
    write(filename=args.output, images=atoms_ortho, format=args.output_format)
    print("done")    
    
    return

#---------------------------------------#
if __name__ == "__main__":
    main()
