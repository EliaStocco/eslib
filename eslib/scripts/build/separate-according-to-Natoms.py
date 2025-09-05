#!/usr/bin/env python
import numpy as np
import os
from eslib.classes.atomic_structures import AtomicStructures
from eslib.input import slist
from eslib.formatting import warning, error, esfmt
from eslib.functions import map2unique

#---------------------------------------#
# Description of the script's purpose
description = "Remove structures containing some chemical species from a file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str  , required=True , help="input file")
    parser.add_argument("-if", "--input_format" , **argv, type=str  , required=False, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"       , **argv, type=str  , required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str  , required=False, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #-------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)
    
    #-------------------#
    Natoms_all = [ len(atoms) for atoms in structures ]
    Natoms = map2unique(Natoms_all)
    print("\tSet of number of atoms in the file: ")
    for n,ii in Natoms.items():
        print(f"\t - {n}: {len(ii)} elements")

    #-------------------#
    # output
    print()
    filename, file_extension = os.path.splitext(args.output)
    for n,ii in Natoms.items():
        file = f"{filename}.N={n}{file_extension}"
        print(f"\tWriting atomic structures with {n} atoms to file '{file}' ... ", end="")
        structures_subset = structures.subsample(ii)
        structures_subset.to_file(file=file,format=args.output_format)
        print("done")
    
#---------------------------------------#     
if __name__ == "__main__":
    main()

