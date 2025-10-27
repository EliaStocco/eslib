#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
from eslib.input import str2bool


#---------------------------------------#
# Description of the script's purpose
description = "Reverse some indices."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , **argv, required=True , type=str     , help="txt input file")
    parser.add_argument("-s", "--sort"  , **argv, required=False, type=str2bool, help="sort the output indices (default: %(default)s)", default=True)
    parser.add_argument("-o", "--output", **argv, required=True , type=str     , help="txt output file")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading indices from input file '{:s}' ... ".format(args.input), end="")
    indices = np.loadtxt(args.input, dtype=int)
    print("done")
    print("\tNumber of indices read: {:d}".format(len(indices)))
    non_null = np.sum(indices != -1)
    print("\tNumber of non-null indices: {:d}".format(non_null))
    
    #------------------#
    print("\tReversing indices ... ", end="")
    reverted_indices = np.arange(len(indices))[indices!=-1]
    print("done")
    assert len(reverted_indices) == non_null, "Number of reverted indices does not match number of non-null indices"
    
    #------------------#
    if args.sort:
        print("\tSorting reverted indices ... ", end="")
        reverted_indices = np.sort(reverted_indices)
        print("done")
    
    #------------------#
    print("\tWriting reversed indices to output file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output, reverted_indices, fmt="%d")
    print("done")
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
