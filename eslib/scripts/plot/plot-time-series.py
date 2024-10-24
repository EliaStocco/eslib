#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from eslib.classes.physical_tensor import PhysicalTensor
from eslib.formatting import esfmt

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
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    data = PhysicalTensor.from_file(file=args.input)

    data = np.atleast_2d(data)

    # Get the number of rows and columns from the array shape
    rows, cols = data.shape
    if rows == 1:
        data = data.T
        cols = rows

    # Create a plot for each row in the array
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


if __name__ == "__main__":
    main()
