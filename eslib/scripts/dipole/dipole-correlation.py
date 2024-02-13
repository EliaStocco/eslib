#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.plot import plot_bisector

#---------------------------------------#
# Description of the script's purpose
description = "Create a correlation plot of the dipole components from two datasets."
closure = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar" : "\b",}
    parser.add_argument("-a" , "--dataset_A", **argv,type=str, help="txt file with the dataset A (DFT)")
    parser.add_argument("-b" , "--dataset_B", **argv,type=str, help="txt file with the dataset B (NN)")
    parser.add_argument("-an", "--name_A"   , **argv,type=str, help="name of the dataset A (default: 'DFT')", default="DFT")
    parser.add_argument("-bn", "--name_B"   , **argv,type=str, help="name of the dataset B (default: 'NN')" , default="NN")   
    parser.add_argument("-o" , "--output"   , **argv,type=str, help="output plot file")
    return parser.parse_args()

#---------------------------------------#
def main():

    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    print("\tReading dataset A ... ",end="")
    A = np.loadtxt(args.dataset_A)
    print("done")
    print("\tdataset A shape: ",A.shape)
    
    print("\n\tReading dataset B ... ",end="")
    B = np.loadtxt(args.dataset_B)
    print("done")
    print("\tdataset B shape: ",B.shape)

    fig,axes = plt.subplots(ncols=3,figsize=(15,5))

    labels = ["x","y","z"]
    for n,ax in enumerate(axes):
        ax.scatter(A[:,n],B[:,n],label=labels[n])
        plot_bisector(ax)
        ax.grid()
        
    axes[1].set_xlabel(args.name_A)
    axes[0].set_ylabel(args.name_B)

    plt.tight_layout()
    print("\tSaving plot to file '{:s}' ... ".format(args.output),end="")
    plt.savefig(args.output)
    print("done")

    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()



print()