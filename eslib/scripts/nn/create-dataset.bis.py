#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.input import str2bool, ilist, slist
from eslib.formatting import esfmt
from eslib.metrics import metrics
from scipy.optimize import minimize
from eslib.show import show_dict
import json

from typing import List
import os
import random

#---------------------------------------#
# Description of the script's purpose
description = "Create a folder with different dataset at varying number of samples with a fixed test dataset."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           , **argv, type=str     , required=True , help="file with the atomic configurations")
    parser.add_argument("-n" , "--incremental_size", **argv, type=int     , required=False, help="incremental size (default: %(default)s)" , default=50)
    parser.add_argument("-t" , "--test_size"       , **argv, type=int     , required=False, help="test dataset size (default: %(default)s)" , default=100)
    parser.add_argument("-o" , "--output"          , **argv, type=str     , required=False, help="output folder (default: %(default)s)", default='data')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")
    print("\tn. of atomic structures: {:d}".format(len(trajectory)))

    #------------------#
    # Create the output folder if it does not exist
    print(f"\tCreating output folder '{args.output}' ... ", end="")
    os.makedirs(args.output, exist_ok=True)
    print("done")
    
    #------------------#
    # test
    indices,test = trajectory.subsample(f":{args.test_size*2}").extract_random(args.test_size)
    assert trajectory.subsample(indices) == test, "coding error"
    assert len(test) == args.test_size, "coding error"
    file = os.path.normpath(f"{args.output}/test.extxyz")
    print(f"\n\tSaving test dataset to file '{file}' ({len(test)} structures) ... ",end="")
    test.to_file(file=file)
    print("done")
    
    #------------------#
    # train
    train = trajectory.remove(indices)
    sizes = np.arange(args.incremental_size,len(train)+1,args.incremental_size)
    # print("\ttrain dataset sizes: ",list(sizes))
    print("\tWriting train datasets:")
    for n,size in enumerate(sizes):
        subsample = train.subsample(":{:d}".format(size))
        file = os.path.normpath(f"{args.output}/train.n={n}.extxyz")
        print(f"\t - {n:2}) size {size:4} --> '{file}' ")
        subsample.to_file(file=file)

#---------------------------------------#
if __name__ == "__main__":
    main()
#