#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.formatting import esfmt
from eslib.classes.trajectory import AtomicStructures
from eslib.plot import generate_colors
from eslib.input import flist

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
    parser.add_argument("-t", "--time"  , type=str, **argv, required=False, help="time keyword (default: %(default)s)", default=None)
    parser.add_argument("-xlim", "--xlim"  , type=flist, **argv, required=False, help="xlim (default: %(default)s)", default=[None,None])
    parser.add_argument("-ylim", "--ylim"  , type=flist, **argv, required=False, help="ylim (default: %(default)s)", default=[None,None])
    parser.add_argument("-s", "--size"  , type=float, **argv, required=False, help="point size (default: %(default)s)", default=1)
    parser.add_argument("-xl", "--xlabel"  , type=str, **argv, required=False, help="xlabel (default: %(default)s)", default="step")
    parser.add_argument("-yl", "--ylabel"  , type=str, **argv, required=False, help="ylabel size (default: %(default)s)", default="")
    parser.add_argument("-o", "--output", type=str, **argv, required=False, help="output file (default: %(default)s)", default=None)
    return parser

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
    time = None
    if args.time is not None:
        print("\tExtracting '{:s}' from the trajectory ... ".format(args.time), end="")
        time = atoms.get(args.time)  
        print("done")

        print("\t'{:s}' shape: ".format(args.time),data.shape)

    #------------------#
    if args.output is None:
        file = "{:s}.pdf".format(args.name)
    else:
        file = args.output

    #------------------#
    # plot
    print("\tSaving plot to file '{:s}'".format(file), end="")
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
            ax.plot(data[:,n], label=str(n+1),marker='o',color=colors[n],markersize=args.size)
        else:
            ax.plot(time,data[:,n], label=str(n+1),marker='o',color=colors[n],markersize=args.size)

    # Add labels and legend
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.xlim(args.xlim[0],args.xlim[1])
    plt.ylim(args.ylim[0],args.ylim[1])
    plt.grid()
    plt.tight_layout()
    plt.legend()

    # Save or show the plot
    if file:
        plt.savefig(file)
    else:
        plt.show()
    print("done")

if __name__ == "__main__":
    main()
