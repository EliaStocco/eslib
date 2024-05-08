#!/usr/bin/env python
import json
import numpy as np
from eslib.input import size_type
from eslib.sklearn_metrics import metrics
from eslib.formatting import esfmt

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
    parser.add_argument("-p", "--predicted", **argv,type=str, help="txt file with the predicted values (default: 'pred.txt')", default="pred.txt")
    parser.add_argument("-e", "--expected" , **argv,type=str, help="txt file with the expected values (default: 'exp.txt')", default="exp.txt")
    parser.add_argument("-m", "--metrics"  , **argv,type=lambda s: size_type(s,dtype=str), help="list of regression metrics (default: ['RMSE','MAE'])" , default=["RMSE","MAE"])
    parser.add_argument("-o", "--output"   , **argv,type=str, help="JSON output file with the computed metrics (default: None)", default=None)
    return parser

@esfmt(prepare_args,description)
def main(args):

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
        print("\n\tSAving results to file '{:s}' ... ".format(args.output), end="")
        with open(args.output, "w") as f:
            json.dump(results, f)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
