#!/usr/bin/env python
import os
import numpy as np
from ase import Atoms
from typing import List
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Split a extxyz on many files with all the structures with the same number of atmoms."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"        , type=str, required=True , **argv, help="input file [au]")
    parser.add_argument("-if" , "--input_format" , type=str, required=False, **argv, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o"  , "--output"       , type=str, required=True , **argv, help="output file")
    parser.add_argument("-of" , "--output_format", type=str, required=False, **argv, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    structures:List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tn. of structures: ",len(structures))
    
    Natoms = structures.call(lambda x: x.get_global_number_of_atoms()) 
    for n in np.unique(Natoms):
        
        file_name, file_extension  = os.path.splitext(args.output)
        file = f"{file_name}.Natoms={n}{file_extension}"
        file = os.path.normpath(file)
        
        print(f"\tSaving structures with {n} atoms to {file} ... ",end="")
        
        ii = np.where(Natoms==n)[0]
        structures_n = structures.subsample(ii)
        
        assert n == structures_n.num_atoms(), "error"        
        
        structures_n.to_file(file=file,format=args.output_format)
        
        print("done")
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
