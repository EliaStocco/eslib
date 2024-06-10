#!/usr/bin/env python
import json
import numpy as np
from eslib.input import size_type
from eslib.sklearn_metrics import metrics
from eslib.formatting import esfmt
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = """Evaluate a regression metric (using sklearn) between two datasets.""" + """The possible metrics are: """ + str(list(metrics.keys()))

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
    parser.add_argument("-m", "--metrics"  , **argv,type=lambda s: size_type(s,dtype=str), help="list of regression metrics (default: %(default)s)" , default=["RMSE"])
    parser.add_argument("-o", "--output"   , **argv,type=str, help="JSON output file with the computed metrics (default: %(default)s)", default=None)
    return parser

@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    if args.input is not None:
        print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
        atoms = AtomicStructures.from_file(file=args.input,format="extxyz")
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
    for n,k in enumerate(args.metrics):
        args.metrics[n] = k.lower()

    if "all" in args.metrics:
        metrics_to_evaluate = list(metrics.keys())
    else:
        metrics_to_evaluate = [k for k in args.metrics]

    print("\n\tMetrics to be evaluated: ",metrics_to_evaluate)
    
    #------------------#
    print("\tEvaluating metrics: ")
    results = dict()
    for k in metrics_to_evaluate:
        func = metrics[k]
        print("\t{:>6} : ".format(k),end="")
        results[k] = func(predicted,expected)
        print("{:>10.6e}".format(results[k]))

    #------------------#
    if args.output is not None:
        print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
        with open(args.output, "w") as f:
            json.dump(results, f)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
