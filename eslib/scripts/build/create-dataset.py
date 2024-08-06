#!/usr/bin/env python
import numpy as np
from classes.atomic_structures import AtomicStructures
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
description = "Create a folder with different dataset at varying number of samples."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           , **argv, type=str     , required=True , help="file with the atomic configurations [a.u]")
    parser.add_argument("-n" , "--incremental_size", **argv, type=int     , required=False, help="incremental size (default: %(default)s)" , default=100)
    parser.add_argument("-N" , "--number_of_files" , **argv, type=int     , required=False, help="max number of files (default: %(default)s)" , default=100)
    parser.add_argument("-in", "--initial_size"    , **argv, type=int     , required=False, help="initial size (default: %(default)s)" , default=100)
    parser.add_argument("-p" , "--percentages"     , **argv, type=ilist   , required=False, help="list with the sizes of the subsamples (default: %(default)s)", default=[80,20])
    parser.add_argument("-nl", "--name_list"       , **argv, type=slist   , required=False, help="list with the names of the subsamples (default: %(default)s)", default=["train","test"])
    parser.add_argument("-sa", "--shuffle_all"     , **argv, type=str2bool, required=False, help="whether to shuffle all the structures (default: %(default)s)", default=False)
    parser.add_argument("-sd", "--shuffle_dataset" , **argv, type=str2bool, required=False, help="whether to shuffle structures within a single dataset (default: %(default)s)", default=True)
    parser.add_argument("-o" , "--output"          , **argv, type=str     , required=False, help="output folder (default: %(default)s)", default='data')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    args.percentages = np.asarray(args.percentages)
    assert args.percentages.sum() == 100, "Sum of percentages must be 100"

    #------------------#
    assert len(args.percentages) == len(args.name_list), "The number of names must be the same as the number of percentages"

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")
    print("\tn. of atomic structures: {:d}".format(len(trajectory)))

     #------------------#
    if args.shuffle_all:
        print("\tShuffling all the structures... ",end="")
        random.shuffle(trajectory)
        print("done")

    #------------------#
    # Create the output folder if it does not exist
    print(f"\tCreating output folder '{args.output}' ... ", end="")
    os.makedirs(args.output, exist_ok=True)
    print("done")

    #------------------#
    sizes = np.arange(args.initial_size, len(trajectory)+1, args.incremental_size)
    if len(sizes) > args.number_of_files:
        sizes = sizes[:args.number_of_files]
    print("\tDataset sizes: ",list(sizes))

    for n,size in enumerate(sizes):

        #------------------#
        print("\n\tSubsampling {:d}/{:d} ... ".format(n+1,len(sizes)),end="")
        subsample = trajectory.subsample(":{:d}".format(size))
        print("done")
        print("\t\ttotal size: {:d}".format(len(subsample)))
        assert len(subsample) == size, "Length of subsample must be equal to size"

        #------------------#
        if args.shuffle_dataset:
            print("\t\tShuffling ... ",end="")
            random.shuffle(subsample)
            print("done")

    
        #------------------#
        single_dataset:List[AtomicStructures] = [None]*len(args.percentages)
        single_sizes = np.asarray(args.percentages*size//100)
        print("\t\tSingle sizes: ",list(single_sizes))
        assert single_sizes.sum() <= size, "Sum of sizes must be less than size"
        k = 0 
        for i,ii in enumerate(single_sizes):
            single_dataset[i] = subsample.subsample("{:d}:{:d}".format(k,k+ii))
            k += ii

        #------------------#
        print("\t\tWriting subsample to files: ")
        for i,name in enumerate(args.name_list):
            file = f"{args.output}/{name}.n={n}.extxyz"
            print("\t\t- {:s} (of size {:d}) --> {:s}".format(name,single_sizes[i],file))
            single_dataset[i].to_file(file=file,format="extxyz")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
#