#!/usr/bin/env python
import json
import numpy as np
from eslib.input import size_type, str2bool
from eslib.metrics import metrics
from eslib.formatting import esfmt, warning
from classes.atomic_structures import AtomicStructures

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

    assert predicted.ndim == expected.ndim 
    assert predicted.shape == expected.shape 

    # if predicted.ndim == 3 and not args.atomic:
    #     print("\n\t{:s}: The provided quantity could be a collection of atomic contribution.\n\tConsider using `-a/--atomic True` for a better estimation of relative metrics (e.g. `relrmse`).".format(warning,))

    #------------------#
    if type(args.metrics) == str:
        args.metrics = [args.metrics]
    
    #------------------#
    for n,k in enumerate(args.metrics):
        args.metrics[n] = k.lower()

    if "all" in args.metrics:
        metrics_to_evaluate = list(metrics.keys())
    else:
        metrics_to_evaluate = [k for k in args.metrics]

    print("\n\tMetrics to be evaluated: ",metrics_to_evaluate)

    
    
    # #------------------#
    # print("\tEvaluating metrics: ")
    # results = dict()
    # for k in metrics_to_evaluate:
    #     func = metrics[k]
    #     print("\t{:>6} : ".format(k),end="")
    #     results[k] = func(predicted,expected)
    #     print("{:>10.6e}".format(results[k]))

    axis = remove_axis(predicted,0)
    # if args.atomic:
    #     axisR = -1 
    # else:
    #     axisR = axisD

    print("\n\tAxes used evaluate the metrics: ",axis)
    # print("\t- for differences/global metrics: ",axisD)
    # print("\t- for reference and norm/atomic metrics: ",axisR)

    print("\n\tEvaluating metrics: ")
    results = {"mean":{}}
    if args.statistics:
        results["std"] = {}
        results["min"] = {}
        results["max"] = {}

    for k in metrics_to_evaluate:
        func = metrics[k]
        print("\t{:<14} : ".format(k),end="")
        tmp:np.ndarray = func(pred=predicted,ref=expected,axis=axis)
        assert tmp.ndim == 1
        results["mean"][k] = tmp.mean()
        if args.statistics:
            results["std"][k] = tmp.std()
            results["min"][k] = tmp.min()
            results["max"][k] = tmp.max()
        # assert np.allclose(tmp.mean(),results["mean"][k])
        print("{:<10.6e}".format(results["mean"][k]))

    if not args.statistics:
        results = results["mean"].copy()

    #------------------#
    if args.output is not None:
        print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
        with open(args.output, "w") as f:
            json.dump(results, f,indent=4)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
