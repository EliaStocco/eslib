#!/usr/bin/env python
import argparse
from copy import copy
import numpy as np
from ase.io import write, read
from typing import Union, List
from eslib.input import union_type
from eslib.formatting import esfmt


# Description of the script's purpose
description = "Subsample an (ASE readable) MD trajectory given a set of indices."


def prepare_args(description):

    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Define command-line arguments

    argv = {"metavar" : "\b"}
    parser.add_argument("-i", "--input",  type=str, default='i-pi.positions_0.xyz', **argv,
                        help="input file containing the MD trajectory (default: 'i-pi.positions_0.xyz')")

    parser.add_argument("-f", "--format", type=str, default='extxyz', **argv, 
                        help="file format (default: extxyz)" )
    
    parser.add_argument("-n", "--indices", type=lambda s: union_type(s,Union[str,List[int]]), **argv, default='indices.txt',
                        help="txt file with the subsampling indices, or list of integers (default: 'indices.txt')")

    parser.add_argument("-o", "--output", type=str, **argv, 
                        help="output file")



    return parser# .parse_args()

@esfmt(prepare_args,description)
def main(args):
   
    print("\tReading atomic structures from file '{:s}' using the 'ase.io.read' with format '{:s}' ... ".format(args.input,args.format), end="")
    atoms = read(args.input,format=args.format,index=":")
    print("done")

    if type(args.indices) == str:
        print("\tReading subsampling indices from file '{:s}' ... ".format(args.indices), end="")
        indices = np.loadtxt(args.indices).astype(int)
        indices.sort()
        print("done")
    else:
        print("\tSubsampling indices: ",args.indice)
        indices = np.asarray(args.indices).astype(int)
        indices.sort()
        print("done")

    print("\tSubsampling atomic structures ... ".format(args.indices), end="")
    new_atoms = [None]*len(indices)
    for n,i in enumerate(indices):
        atoms[i].calc = None # atoms[i].set_calculator(None)
        new_atoms[n] = copy(atoms[i])
    # atoms = list(np.array(atoms,dtype=object)[indices])
    print("done")

    # Write the data to the specified output file with the specified format
    print("\tWriting subsampled atomic structures to file '{:s}' with format '{:s}' ... ".format(args.output, args.format), end="")
    try:
        write(args.output, new_atoms, format=args.format) # fmt)
        print("done")
    except Exception as e:
        print(f"\n\tError: {e}")

if __name__ == "__main__":
    main()