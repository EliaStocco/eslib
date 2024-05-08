#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
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
    return parser# .parse_args()

#---------------------------------------#
def plot_array(input_file, output_file):
    # Load the numpy array from the input file
    data = np.atleast_2d(np.loadtxt(input_file))

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
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    # Call the function with the provided arguments
    plot_array(args.input, args.output)


if __name__ == "__main__":
    main()
