#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.sklearn_metrics import metrics
from eslib.plot import histogram

#---------------------------------------#
# Description of the script's purpose
description = "Detect outliers depending of the RMSE of an atomic structure property."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"              , **argv, type=str  , required=True , help="extxyz file with the atomic configurations [a.u]")
    parser.add_argument("-rn" , "--ref_name"           , **argv, type=str  , required=True , help="name of the reference quantity")
    parser.add_argument("-pn" , "--pred_name"          , **argv, type=str  , required=True , help="name of the predicted quantity")
    parser.add_argument("-t"  , "--threshold"          , **argv, type=float, required=False, help="RMSE threshold (default: 1e-1)", default=1e-1)   
    parser.add_argument("-d"  , "--distribution"       , **argv, type=str  , required=False, help="*.pdf file with the distribution of the RMSE values (default: 'rmse.pdf')", default='rmse.pdf')
    parser.add_argument("-oi" , "--output_indices"     , **argv, type=str  , required=False, help="*.txt output file with the indices of the outliers (default: None)", default=None)
    parser.add_argument("-ogi", "--output_good_indices", **argv, type=str  , required=False, help="*.txt output file with the indices of the non-outliers (default: None)", default=None)
    parser.add_argument("-o"  , "--output"             , **argv, type=str  , required=False, help="*.extxyz output file with the outliers atomic configurations (default: 'outliers.extxyz')", default="outliers.extxyz")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")
    print("\tn. of atomic structures: ",len(trajectory),end="\n\n")

    #------------------#
    assert trajectory.is_there(args.ref_name,where="info")
    assert trajectory.is_there(args.pred_name,where="info")
    assert args.threshold > 0.

    #------------------#
    print("\tExtracting '{:s}' from the atomic structures... ".format(args.ref_name), end="")
    real = trajectory.get_info(args.ref_name)
    print("done")
    print("\t'{:s}' shape: ".format(args.ref_name),real.shape,end="\n\n")

    #------------------#
    print("\tExtracting '{:s}' from the atomic structures... ".format(args.pred_name), end="")
    pred = trajectory.get_info(args.pred_name)
    print("done")
    print("\t'{:s}' shape: ".format(args.pred_name),pred.shape,end="\n\n")

    #------------------#
    print("\tComputing RMSE ... ", end="")
    norm_fun:callable = metrics["norm"]
    rmse = norm_fun(real,pred,axis=1)
    assert rmse.shape[0] ==  real.shape[0]
    print("done")

    #------------------#
    print("\tDetecting outliers ... ", end="")
    booleans = rmse > args.threshold
    indices = np.where(booleans)[0]
    tot = len(indices)
    print("done")

    print("\tN. of found outliers: {:d}".format(tot),end="\n\n")

    #------------------#
    if args.distribution is not None:
        print("\tSaving the RMSE distribution to file '{:s}' ... ".format(args.distribution), end="")
        histogram(rmse,args.distribution)
        print("done")

    #------------------#
    outliers = trajectory.subsample(indices)

    print("\tSaving outliers to file '{:s}' ... ".format(args.output), end="")
    outliers.to_file(args.output)
    print("done")

    if args.output_indices is not None:
        print("\tSaving indices of the outlier structures to file '{:s}' ... ".format(args.output_indices), end="")
        np.savetxt(args.output_indices,indices.astype(int))
        print("done")

    if args.output_good_indices is not None:
        print("\tSaving indices of the good structures to file '{:s}' ... ".format(args.output_good_indices), end="")
        good_indices = np.delete(np.arange(len(rmse)), indices)
        np.savetxt(args.output_good_indices,good_indices.astype(int))
        print("done")

    
#---------------------------------------#
if __name__ == "__main__":
    main()