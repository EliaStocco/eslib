#!/usr/bin/env python
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, float_format, eslog
from eslib.input import ilist, str2bool
from eslib.tools import get_files
# from eslib.classes.physical_tensor import PhysicalTensor
import concurrent.futures
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Count number of snapshots with NaN in a property."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str     , help="input file [extxyz]")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k"  , "--keyword"      , **argv, required=True , type=str     , help="keyword of the info/array")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    with eslog(f"Reading atomic structures from file '{args.input}'"):
        trajectory = AtomicStructures.from_file(file=args.input,format=args.input_format)
    N = len(trajectory)
    print("\tn. of atomic structures: ",N)
    
    prop = trajectory.get(args.keyword)
    n_nan = 0
    for n, val in enumerate(prop):
        if np.any(np.isnan(val)):
            n_nan += 1
            print(f"\tSnapshot {n} has NaN in '{args.keyword}'")
    print(f"\n\tTotal number of snapshots with NaN in '{args.keyword}': {n_nan} out of {N} --> ({N-n_nan}/{N} have the keyword)")

#---------------------------------------#
if __name__ == "__main__":
    main()

