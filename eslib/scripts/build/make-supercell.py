#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from eslib.formatting import matrix2str
from ase.build import make_supercell
from scipy.spatial.transform import Rotation
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Create a supercell for the given atomic structures."

#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, type=str, help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--matrix"       , **argv, type=str, help="txt file with the 3x3 transformation matrix")
    parser.add_argument("-o" , "--output"       , **argv, type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str, help="output file format (default: %(default)s)", default=None)
    return  parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #-------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #-------------------#
    print("\tReading transformation matrix from file '{:s}' ... ".format(args.matrix), end="")
    matrix = np.loadtxt(args.matrix)
    print("done")

    #-------------------#
    print("\n\tCreating the supercells ... ", end="")
    supercell = [None] * len(structures)
    for n,atoms in enumerate(structures):
        supercell[n] = make_supercell(atoms,matrix,wrap=False)
    supercell = AtomicStructures(supercell)
    print("done")

    #-------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    supercell.to_file(file=args.output,format=args.output_format)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()
