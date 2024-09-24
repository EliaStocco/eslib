#!/usr/bin/env python
import numpy as np
from ase.io import write
from classes.atomic_structures import AtomicStructures
from eslib.input import slist
from eslib.formatting import warning, error, esfmt
from eslib.classes.physical_tensor import PhysicalTensor

#---------------------------------------#
# Description of the script's purpose
description = "Remove species for a file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str  , required=True , help="input file")
    parser.add_argument("-if", "--input_format" , **argv, type=str  , required=False, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--species"      , **argv, type=slist, required=True , help="chemical species to be removed")
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
    # remove species
    print("\n\tRemoving species: ",args.species)
    for n,atoms in enumerate(structures):
        print("\t - {:d}/{:d}".format(n+1,N),end="\r")
        symbols = atoms.get_chemical_symbols()
        indices = [ i for i,s in enumerate(symbols) if s not in args.species ]
        # if len(indices) != len(atoms):
        #     pass
        structures[n] = atoms[indices]
    print("\tdone")
        
    #-------------------#
    # output
    print("\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
    
#---------------------------------------#     
if __name__ == "__main__":
    main()

