#!/usr/bin/env python
import numpy as np
from typing import Union, List
from eslib.input import union_type
from eslib.formatting import esfmt
from classes.atomic_structures import AtomicStructures
from eslib.input import str2bool

# Description of the script's purpose
description = "Subsample an (ASE readable) MD trajectory given a set of indices."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    intype = lambda s: union_type(s,Union[str,List[int]])
    argv = {"metavar" : "\b"}
    parser.add_argument("-i" , "--input"        , required=True , **argv, type=str     , help="input file")
    parser.add_argument("-if", "--input_format" , required=False, **argv, type=str     , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--indices"      , required=False, **argv, type=intype  , help="txt file with the subsampling indices, or list of integers (default: %(default)s)",default='indices.txt')
    parser.add_argument("-s" , "--sort"         , required=False, **argv, type=str2bool, help="sort indices (default: %(default)s)", default=False)
    parser.add_argument("-o" , "--output"       , required=True , **argv, type=str     , help="output file")
    parser.add_argument("-of", "--output_format", required=False, **argv, type=str     , help="output file format (default: %(default)s)", default=None)
    return parser

@esfmt(prepare_args,description)
def main(args):
   
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")

    if type(args.indices) == str:
        print("\tReading subsampling indices from file '{:s}' ... ".format(args.indices), end="")
        indices = np.loadtxt(args.indices).astype(int)
        print("done")
    else:
        print("\tSubsampling indices: ",args.indice)
        indices = np.asarray(args.indices).astype(int)
        print("done")

    if args.sort:
        print("\tSorting indices: ",end="")
        indices.sort()
        print("done")

    print("\tSubsampling atomic structures ... ".format(args.indices), end="")
    # new_atoms = [None]*len(indices)
    # for n,i in enumerate(indices):
    #     atoms[i].calc = None # atoms[i].set_calculator(None)
    #     new_atoms[n] = copy(atoms[i])
    # # atoms = list(np.array(atoms,dtype=object)[indices])
    new_atoms = atoms.subsample(indices)
    print("done")

    # Write the data to the specified output file with the specified format
    print("\tWriting subsampled atomic structures to file '{:s}' ... ".format(args.output), end="")
    new_atoms.to_file(file=args.output, format=args.output_format)
    print("done")

if __name__ == "__main__":
    main()