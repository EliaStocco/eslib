#!/usr/bin/env python
import numpy as np
from ase.io import read
from eslib.classes.dipole import DipoleModel
# from elia.classes.trajectory import trajectory as Trajectory

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate the dipole of a set of atomic structures according to (previously created) linear model."
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA  + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    information     = Fore.YELLOW   + Style.NORMAL + information             + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv,type=str, help="file with the atomic configurations [a.u]")
    parser.add_argument("-m", "--model"    , **argv,type=str, help="pickle file with the dipole linear model (default: 'DipoleModel.pickle')", default='DipoleModel.pickle')
    parser.add_argument("-f", "--frame"    , **argv,type=str, help="frame type [eckart,global] (default: global)", default="global", choices=["eckart", "global"])
    parser.add_argument("-o", "--output"   , **argv,type=str, help="output file with the dipole values (default: 'dipole.linear-model.txt')", default="dipole.linear-model.txt")
    return parser# .parse_args()

#---------------------------------------#
def main():

    #------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = read(args.input,index=":")
    print("done")

    #------------------#
    # linear model
    print("\tLoading the dipole linear model from file '{:s}' ... ".format(args.model), end="")
    model = DipoleModel.from_pickle(args.model)
    print("done")

    #------------------#
    # dipole
    print("\tComputing the dipoles using the linear model using the '{:s}' frame ... ".format(args.frame), end="")
    dipoles = model.get(trajectory,frame=args.frame)
    print("done")

    #------------------#
    # output
    print("\n\tSaving the dipoles to file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output,dipoles)
    print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()