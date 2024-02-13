#!/usr/bin/env python
import argparse
import numpy as np
from ase.io import read
import pandas as pd

#---------------------------------------#
# Description of the script's purpose
description = "Read a trajectory from an extxyz file, then extract a property or array and save it to file."
warning = "***Warning***"
closure = "Job done :)"
keywords = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"
#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN  + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass
#---------------------------------------#
def prepare_args():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-t",  "--trajectory",  type=str, **argv, help="input extxyz file [a.u.] (default: 'trajectory.extxyz')", default="trajectory.extxyz")
    parser.add_argument("-pm", "--phonon_modes",type=str, **argv, help="phonon modes file computed by 'post-process-phonopy.py' (default: 'phonon-modes.pickle')", default="phonon-modes.pickle")
    return parser.parse_args()

def main():

    #---------------------------------------#
    # print the script's description
    print("\n\t{:s}".format(description))

    #---------------------------------------#
    # parse the user input
    args = prepare_args()
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #---------------------------------------#
    # read trajectory
    print("\tReading trajectory from file '{:s}' ... ".format(args.trajectory), end="")
    trajectory = read(args.trajectory,format="extxyz",index=":")
    print("done")

    #---------------------------------------#
    # read phonon modes ('phonon-modes.pickle')
    print("\tReading phonon modes from file '{:s}' ... ".format(args.phonon_modes), end="")
    pm = pd.read_pickle(args.phonon_modes)
    print("done")

    #---------------------------------------#



    
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()