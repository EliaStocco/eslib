#!/usr/bin/env python
from ast import arg
import matplotlib.pyplot as plt
import numpy as np

from eslib.classes.physical_tensor import PhysicalTensor
from eslib.formatting import esfmt
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Plot a time series from a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input", type=str, **argv, required=True, help='input txt file')
    parser.add_argument("-o","--output", type=str, **argv, required=True, help='output file for the plot')
    parser.add_argument("-r","--remove_mean", type=str2bool, **argv, required=False, help='remove_mean (default: %(default)s)', default=False)
    parser.add_argument("-a","--axis", type=int, **argv, required=False, help='axis (default: %(default)s)', default=0)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\tReading data from file '{:s}' ... ".format(args.input), end="")
    data = PhysicalTensor.from_file(file=args.input)
    print("done")

    data = np.atleast_2d(data)
    print("\tdata shape: ",data.shape)
    
    if args.remove_mean:
        print("\tRemoving mean ... ",end="")
        data -= np.mean(data,axis=args.axis,keepdims=True)
        print("done")

    # Get the number of rows and columns from the array shape
    rows, cols = data.shape
    if rows == 1:
        data = data.T
        cols = rows

    # Create a plot for each row in the array
    print(f"\tSaving plot to file '{args.output}' ... ", end="")
    fig,ax = plt.subplots(figsize=(15,5))
    for n in range(cols):
        ax.plot(data[:,n], label=str(n+1),marker='o')

    # Add labels and legend
    plt.xlabel('time/row index')
    plt.grid()
    plt.tight_layout()
    plt.legend()

    # Save or show the plot
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
    print("done")


if __name__ == "__main__":
    main()
