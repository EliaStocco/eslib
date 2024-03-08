#!/usr/bin/env python
import numpy as np
import random
from eslib.formatting import esfmt, error
from eslib.classes.trajectory import trajectory as Trajectory
from eslib.input import ilist, slist, str2bool
from typing import List
from ase import Atoms
from ase.io import write

#---------------------------------------#
# Description of the script's purpose
description = "Extract a set of subsamples from a file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input"        , type=str     , **argv, required=True , help='input file')
    parser.add_argument("-if", "--input_format", type=str     , **argv, required=False, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-n", "--sizes"        , type=ilist   , **argv, required=True , help="list with the sizes of the subsamples")
    parser.add_argument("-s", "--shuffle"      , type=str2bool, **argv, required=False, help="whether to shuffle (default: true)", default=True)
    parser.add_argument("-o", "--output"       , type=slist   , **argv, required=True , help="output files")
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory:List[Atoms] = list(Trajectory(args.input,format=args.input_format,index=":"))
    print("done")

    #------------------#
    sizes = np.asarray(args.sizes)
    assert sizes.sum() <= len(trajectory)

    #------------------#
    if args.shuffle:
        print("\tShuffling ... ",end="")
        random.shuffle(trajectory)
        print("done")

    #------------------#
    print("\tSubsampling ... ",end="")
    subsamples = [None]*len(sizes)
    k = 0 
    for i,n in enumerate(sizes):
        subsamples[i] = trajectory[k:k+n]
        k += n
    print("done")

    #------------------#
    print("\n\tWriting subsamples to file: ")
    for n,(sample,file) in enumerate(zip(subsamples,args.output)):
        print("\t\t{:d}) '{:s}' ... ".format(n,file),end="")
        try:
            write(file, sample) # fmt)
            print("done")
        except Exception as e:
            print(f"\n\t{error}: {e}")

if __name__ == "__main__":
    main()
