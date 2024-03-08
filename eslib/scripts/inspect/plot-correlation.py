#!/usr/bin/env python
import matplotlib.pyplot as plt
from eslib.plot import plot_bisector
from eslib.formatting import esfmt
from eslib.classes.trajectory import trajectory, info

#---------------------------------------#
# Description of the script's purpose
description = "Plot the correlation plot between two info of a extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input"     , type=str, **argv, required=True , help='input extxyz file')
    parser.add_argument("-a" , "--keyword_A", type=str, **argv, required=True , help="property A keyword")
    parser.add_argument("-b" , "--keyword_B", type=str, **argv, required=True , help="property B keyword")
    parser.add_argument("-o", "--output"    , type=str, **argv, required=False, help="output file (default: 'corr.pdf')", default='corr.pdf')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = trajectory(args.input,format="extxyz")
    print("done")

    #------------------#
    # data
    print("\tExtracting '{:s}' from the trajectory ... ".format(args.keyword_A), end="")
    A = info(atoms,args.keyword_A)  
    print("done")
    print("\t'{:s}' shape: ".format(args.keyword_A),A.shape)

    print("\tExtracting '{:s}' from the trajectory ... ".format(args.keyword_B), end="")
    B = info(atoms,args.keyword_B)  
    print("done")
    print("\t'{:s}' shape: ".format(args.keyword_B),B.shape)

    #------------------#
    assert A.shape == B.shape

    #------------------#
    if A.shape[1] == 3:
        fig,axes = plt.subplots(ncols=3,figsize=(15,5))

        labels = ["x","y","z"]
        for n,ax in enumerate(axes):
            ax.scatter(A[:,n],B[:,n],label=labels[n])
            plot_bisector(ax)
            ax.grid()
            
        axes[1].set_xlabel(args.keyword_A)
        axes[0].set_ylabel(args.keyword_B)

        plt.tight_layout()
        print("\tSaving plot to file '{:s}' ... ".format(args.output),end="")
        plt.savefig(args.output)
        print("done")

    else:
        raise ValueError("not implemented yet")

if __name__ == "__main__":
    main()
