#!/usr/bin/env python
import json, os
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import warning, esfmt
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool
from eslib.show import show_dict


#---------------------------------------#
# Description of the script's purpose
description = "Add structure info as a tag to an extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-t" , "--tag"         , **argv, type=str, required=True , help="tah name (default: %(default)s)", default="index")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)

    #---------------------------------------#
    print("\tStoring tags ... ", end="")
    name = f"tag:{args.tag}"
    v = np.arange(len(structures))
    structures.set(name,np.full(N,v),"info")
    print("done")
    
    #---------------------------------------#
    print("\n\tWriting trajectory to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

if __name__ == "__main__":
    main()

