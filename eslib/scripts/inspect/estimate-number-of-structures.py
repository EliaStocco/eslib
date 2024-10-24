#!/usr/bin/env python
import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Estimate the number of structure contained in an 'extxyz' file in an efficient way."

#---------------------------------------#
def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)
    
#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input", **argv, type=str, help="input file")
    # parser.add_argument("-if", "--input_format", **argv, type=str, help="input file format (default: %(default)s)", default=None)
    return parser  # .parse_args()

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #---------------------------------------#
    args.input = str(args.input)
    if ( not args.input.endswith("extxyz") ) and ( not args.input.endswith("xyz")):
        text = "Input file can only be 'extxyz' or 'xyz'."
        raise ValueError(text)

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input, format='extxyz',index=0)[0]
    print("done")
    Natoms = atoms.get_global_number_of_atoms()
    print("\tn. of atoms: {:d}".format(Natoms))

    #---------------------------------------#
    Nlines = count_lines(args.input)
    print("\n\tn. lines in input file: ",Nlines)

    Nstructures = int( Nlines / (Natoms+2) )
    print("\n\tn. of structures: ",Nstructures)

#---------------------------------------#
if __name__ == "__main__":
    main()
