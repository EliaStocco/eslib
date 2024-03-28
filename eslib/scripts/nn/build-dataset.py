#!/usr/bin/env python
from ase.io import read
import random
import os
import torch
import numpy as np
from copy import copy
from eslib.functions import str2bool
from eslib.nn.dataset import make_dataset
from eslib.input import size_type
from eslib.formatting import esfmt

#---------------------------------------#
description = "Build a dataset from an 'extxyz' file, readable by 'train-e3nn-model.py'."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i", "--input"  , type=str, **argv, help="input 'extxyz' file with the atomic structures")
    parser.add_argument("-o", "--output" , type=str, **argv, default="dataset", help="prefix for the output files (default: 'dataset')")
    parser.add_argument("-n", "--size"   , type=size_type, **argv, help="size of the train, val, and test datasets (example: '[1000,100,100]')",default=np.asarray([1000,100,100]))
    parser.add_argument("-r", "--random", type=str2bool, **argv, default=True, help="whether the atomic structures are chosen randomly (default: true)")
    parser.add_argument("-s", "--seed", type=int, **argv, default=None, help="seed of the random numbers generator (default: None)")
    parser.add_argument("-pbc", "--pbc"  ,  type=str2bool, **argv, default=True, help="whether the system is periodic (default: True)")
    parser.add_argument("-rc", "--cutoff_radius",  type=float, **argv, help="cutoff radius in atomic unit")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    args.size = np.asarray(args.size)    
    if np.any( args.size[1:] < 0 ):
        raise ValueError("The size of the datasets should be non-negative.")
    if args.cutoff_radius <= 0 :
        raise ValueError("The cutoff radius (-rc,--cutoff_radius) has to be positive.")
    
    # Print the script's description
    print("\n\t{:s}\n".format(description))

    # Print the atomic structures
    print("\tReading atomic structures from file '{:s}' using the 'ase.io.read' ... ".format(args.input), end="")
    atoms = read(args.input,format="extxyz",index=":")
    print("done")

    #------------------#
    if args.random:
        if args.seed is not None:
            print("\tSetting the seed of the random numbers generator equal to {:d} ... ".format(args.seed), end="")
            random.seed(args.seed)
            print("done")
        print("\tShuffling the atomic structures ... ", end="")
        random.shuffle(atoms)
        print("done")

    #------------------#
    if args.size[0] == -1:
        args.size = np.asarray([len(atoms),0,0])
    N = args.size.sum()
    print("\tExtracting {:d} atomic structures ... ".format(N), end="")
    atoms = atoms[:N]
    print("done")

    n,i,j = args.size
    dataset = {
        "train" : copy(atoms[:n]),
        "val"   : copy(atoms[n:n+i]),
        "test"  : copy(atoms[n+i:n+i+j]),
    }

    #------------------#
    print()
    for k in dataset.keys():
        print("\tBuilding the '{:s}' dataset ({:d} atomic structures) ... ".format(k,len(dataset[k])), end="")
        dataset[k] = make_dataset(systems=dataset[k],max_radius=args.cutoff_radius,disable=True)   
        print("done")

    #------------------#
    example = dataset['train'][0].to_dict()
    keys = example.keys()
    print("\n\tStored information:")
    for k in keys:
        try: shape = list(example[k].shape)
        except: shape = str(type(example[k]))
        print("\t\t{:20s}: ".format(k),shape)        

    #------------------#
    print()
    for k in dataset.keys():
        file = "{:s}.{:s}.pth".format(args.output,k)
        file = os.path.normpath(file)
        d = dataset[k]
        print("\tSaving the '{:s}' dataset to file '{:s}' ... ".format(k,file), end="")
        torch.save(d,file)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
