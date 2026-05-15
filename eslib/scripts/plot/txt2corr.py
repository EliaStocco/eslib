#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.plot import plot_bisector

#---------------------------------------#
# Description of the script's purpose
description = "Plot the correlation plot between two info of a extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a" , "--file_A", type=str, **argv, required=True , help="txt file A")
    parser.add_argument("-b" , "--file_B", type=str, **argv, required=True , help="txt file A")
    parser.add_argument("-s", "--size"  , type=float, **argv, required=False, help="point size (default: %(default)s)", default=1)
    parser.add_argument("-o", "--output"    , type=str, **argv, required=False, help="output file (default: %(default)s)", default='corr.pdf')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):



    #------------------#
    # data
    print("\tExtracting '{:s}' from the trajectory ... ".format(args.file_A), end="")
    A = np.loadtxt(args.file_A)
    print("done")
    print("\tA.shape: ",A.shape)

    print("\tExtracting '{:s}' from the trajectory ... ".format(args.file_B), end="")
    B = np.loadtxt(args.file_B)
    print("done")
    print("\tB.shape: ",B.shape)
    
    A = A.flatten()
    B = B.flatten()

    #------------------#
    assert A.shape == B.shape

    #------------------#
    fig,ax = plt.subplots(figsize=(5,5))
    ax.scatter(A,B,s=args.size)
    # plot_bisector(ax)
    ax.grid()
    ax.set_xlabel(args.file_A)
    ax.set_ylabel(args.file_B)

    plt.tight_layout()
    print("\tSaving plot to file '{:s}' ... ".format(args.output),end="")
    plt.savefig(args.output)
    print("done")
    

if __name__ == "__main__":
    main()
