#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.formatting import esfmt
from classes.atomic_structures import AtomicStructures, info
from eslib.plot import generate_colors
from eslib.plot import histogram

#---------------------------------------#
# Description of the script's purpose
description = "Plot a time series of a 'info' of a 'extxyz' file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , type=str, **argv, required=True , help='input extxyz file')
    parser.add_argument("-n", "--name"  , type=str, **argv, required=True , help="info keyword to be plotted")
    parser.add_argument("-o", "--output", type=str, **argv, required=False, help="output file (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
def plot_array(data, output_file, time=None):
    # Load the numpy array from the input file
    data = np.atleast_2d(data)

    # Get the number of rows and columns from the array shape
    rows, cols = data.shape
    if rows == 1:
        data = data.T
        cols = rows

    # Create a plot for each row in the array
    fig,ax = plt.subplots(figsize=(15,5))
    colors = generate_colors(cols,"viridis")
    for n in range(cols):
        if time is None:
            ax.plot(data[:,n], label=str(n+1),marker='o',color=colors[n])
        else:
            ax.plot(time,data[:,n], label=str(n+1),marker='o',color=colors[n])

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

    #------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input)
    print("done")

    #------------------#
    # data
    print("\tExtracting '{:s}' from the trajectory ... ".format(args.name), end="")
    data = atoms.get(args.name)  
    print("done")

    print("\t'{:s}' shape: ".format(args.name),data.shape)

    #------------------#
    # plot
    print("\tSaving plot to file '{:s}'".format(args.output), end="")
    histogram(data,args.output)
    print("done")

if __name__ == "__main__":
    main()
