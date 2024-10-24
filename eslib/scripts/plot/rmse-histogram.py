#!/usr/bin/env python
import numpy as np

from eslib.classes import Trajectory
from eslib.formatting import esfmt
from eslib.plot import histogram

metrics = {
    "rmse" : lambda x,y: np.linalg.norm(x-y,axis=1),
    "relrmse" : lambda x,y: np.linalg.norm(x-y,axis=1)/np.linalg.norm(x,axis=1),
}

#---------------------------------------#
# Description of the script's purpose
description = "Plot the histogram of tha metric between two info."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i", "--input"     , type=str, **argv, required=False , help='input extxyz file (default: %(default)s)',default=None )
    parser.add_argument("-e", "--expected" , **argv,type=str, help="keyword or txt file with the expected values (default: %(default)s)", default="exp.txt")
    parser.add_argument("-p", "--predicted", **argv,type=str, help="keyword or txt file with the predicted values (default: %(default)s)", default="pred.txt")
    parser.add_argument("-m", "--metrics"  , **argv,type=str, help="list of regression metrics (default: %(default)s)" , default="rmse",choices=["rmse","relrmse"])
    parser.add_argument("-o", "--output"   , **argv,type=str, help="pdf output file with the computed metrics (default: %(default)s)", default=None)
    return parser

@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    if args.input is not None:
        print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
        atoms = Trajectory.from_file(file=args.input)
        print("done")

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

    #------------------#
    metrics_to_evaluate = str(args.metrics).lower()
    print("\n\tMetrics to be evaluated: ",metrics_to_evaluate)
    
    #------------------#
    print("\tComputing metric ... ",end="")
    rmse = metrics[metrics_to_evaluate](predicted,expected)
    print("done")

    #------------------#
    print("\tSaving the metric distribution to file '{:s}' ... ".format(args.output), end="")
    histogram(rmse,args.output)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()
