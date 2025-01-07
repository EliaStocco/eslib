#!/usr/bin/env python
import json

import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.input import size_type, str2bool
from eslib.metrics import metrics

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate a regression metric (using sklearn) between two datasets." 
# + """The possible metrics are: """ + str(list(metrics.keys()))

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i", "--input"     , **argv, type=str                             , required=False , help='input extxyz file (default: %(default)s)',default=None )
    parser.add_argument("-e", "--expected"  , **argv, type=str                             , required=False , help="keyword or txt file with the expected values (default: %(default)s)", default="exp.txt")
    parser.add_argument("-p", "--predicted" , **argv, type=str                             , required=False , help="keyword or txt file with the predicted values (default: %(default)s)", default="pred.txt")
    parser.add_argument("-m", "--metrics"   , **argv, type=lambda s: size_type(s,dtype=str), required=False , help="list of regression metrics (default: %(default)s)" , default=["RMSE"])
    # parser.add_argument("-a", "--atomic"    , **argv, type=str2bool                        , required=False , help="whether the quantity has to be interpreted as a collection of atomic contribution (default: %(default)s)" , default=False)
    parser.add_argument("-s", "--statistics", **argv, type=str2bool                        , required=False , help="whether to evaluate the statistics of the metrics (default: %(default)s)", default=False)
    parser.add_argument("-o", "--output"    , **argv, type=str                             , required=False , help="JSON output file with the computed metrics (default: %(default)s)", default=None)
    return parser

#------------------#
def remove_axis(arr:np.ndarray,axis:int)->tuple:
    num_dims = tuple([i for i in range(arr.ndim)])
    return num_dims[:axis] + num_dims[axis+1:]

@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    if args.input is not None:
        print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
        atoms = AtomicStructures.from_file(file=args.input,format="extxyz")
        print("done")
        Natoms = atoms[0].get_global_number_of_atoms()

        predicted = atoms.get(args.predicted)
        expected = atoms.get(args.expected)

    else:
        #------------------#
        print("\tReading predicted values from file '{:s}' ... ".format(args.predicted), end="")
        predicted = np.loadtxt(args.predicted)
        print("done")
        

        #------------------#
        print("\tReading expected values from file '{:s}' ... ".format(args.expected), end="")
        expected = np.loadtxt(args.expected)
        print("done")
        

    print("\tpredicted.shape: ",predicted.shape)
    print("\texpected.shape: ",expected.shape)

    assert predicted.ndim == 2, "the predicted values must be a 2D array"
    assert expected.ndim == 2, "the expected values must be a 2D array"
    assert predicted.ndim == expected.ndim 
    assert predicted.shape == expected.shape 
    
    #------------------#
    print("\tComputing metrics ... ", end="")
    diff  = predicted - expected        
    err_atoms = diff/Natoms
    err_atoms_2 = np.square(err_atoms)
    diff2 = err_atoms_2.sum(axis=1)      # sum over x,y,z
    mean = np.mean(diff2)                # mean over snapshots
    rmse = np.sqrt(mean)
    print("done")
    
    print("\tRMSE (x1000): ",rmse*1000)

    
#---------------------------------------#
if __name__ == "__main__":
    main()
