#!/usr/bin/env python
import numpy as np
from eslib.classes.physical_tensor import PhysicalTensor
import matplotlib.pyplot as plt
import json
from eslib.tools import string2function
from eslib.formatting import esfmt
from scipy.stats import bootstrap

#---------------------------------------#
# Description of the script's purpose
description = "Bootstrap a time series."

N_RESAMPLES = 1000

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, type=str, help="txt/npy input file")
    # parser.add_argument("-a", "--axis", **argv, required=False, type=int, help="axis along compute the bootstrap (default: %(default)s)", default=0)
    parser.add_argument("-c", "--confidence", **argv, required=False, type=float, help="confidence level (default: %(default)s)", default=0.95)
    # parser.add_argument("-f", "--function", **argv, required=False, type=str, help="numpy function or source code (default: %(default)s)", default='mean')
    # parser.add_argument("-o", "--output", **argv, required=False, type=str, help="txt/npy output file (default: %(default)s)", default='hist.txt')
    # parser.add_argument("-p", "--plot", **argv, required=False, type=str, help="plot (default: %(default)s)", default='hist.pdf')
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    data: np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    data = np.atleast_2d(data)
    print("\tdata shape: ", data.shape)

    #------------------#
    # mean
    print("\n\tBootstrapping (mean)  ... ",end="")
    data_mean = bootstrap(data=(data,), method='basic', axis=0, n_resamples=N_RESAMPLES, statistic=np.mean, confidence_level=args.confidence)
    print("done")

    #------------------#
    values = {
        "ci_low": list(data_mean.confidence_interval.low),
        "ci_high": list(data_mean.confidence_interval.high),
        "se": list(data_mean.standard_error),
        "mean": list(np.mean(data,axis=0)),
    }
    
    print("\tBootstrap confidence interval: ")
    print("\t -  low: ", values["ci_low"])
    print("\t - high: ", values["ci_high"])
    print("\tBootstrap standard error: ", values["se"])
    print("\tMean: ", values["mean"])

    with open("mean.json", "w") as f:
        json.dump(values, f,indent=4)

    #------------------#
    print("\n\tSaving std value to file '{:s}' ... ".format("mean.txt"), end="")
    hist = PhysicalTensor(data_mean.bootstrap_distribution.T)
    hist.to_file(file='mean.txt',header="Average value of the dipole \nColumns correspond to each component of the dipole (1st:x, 2nd:y, 3rd:z)")
    print("done")

    del data_mean

    #------------------#
    # isotropic fluctuation

    # def fluctuation(data:np.ndarray):
    #     # return np.mean(np.square(data).sum(axis=1))-np.square(np.mean(data,axis=0)).sum()
    #     return np.std(data,axis=0)
    
    print("\n\tBootstrapping (fluctuation)  ... ",end="")
    data_std = bootstrap(data=(data,), method='basic', axis=0, n_resamples=N_RESAMPLES, statistic=np.std, confidence_level=args.confidence)
    print("done")

    #------------------#
    values = {
        "ci_low": list(data_std.confidence_interval.low),
        "ci_high": list(data_std.confidence_interval.high),
        "se": list(data_std.standard_error),
        "mean": list(np.std(data,axis=0)),
    }
    

    print("\tBootstrap confidence interval: ")
    print("\t -  low: ", values["ci_low"])
    print("\t - high: ", values["ci_high"])
    print("\tBootstrap standard error: ", values["se"])
    print("\tMean: ", values["mean"])

    with open("std.json", "w") as f:
        json.dump(values, f,indent=4)


    #------------------#
    print("\n\tSaving mean value to file '{:s}' ... ".format("std.txt"), end="")
    hist = PhysicalTensor(data_std.bootstrap_distribution.T)
    hist.to_file(file='std.txt',header="Standard deviation of the dipole \nColumns correspond to each component of the dipole (1st:x, 2nd:y, 3rd:z)")
    print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
