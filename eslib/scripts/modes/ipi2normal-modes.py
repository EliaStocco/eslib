#!/usr/bin/env python
from eslib.classes.normal_modes import NormalModes
# from elia.formatting import matrix2str
# from elia.tools import convert
# from elia.functions import output_folder
# from elia.input import size_type
# from elia.functions import phonopy2atoms
import argparse
import numpy as np
# import yaml
import pandas as pd
# from icecream import ic
# import os
from ase.io import read
# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
# Description of the script's purpose
description = "Prepare the necessary file to project a MD trajectory onto phonon modes: read results from i-PI."
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
    parser.add_argument("-r",  "--reference",       type=str, **argv, 
                        help="reference structure w.r.t. which the phonons are computed [a.u.] (default: None)",default=None)
    parser.add_argument("-f",  "--folder",         type=str, **argv, 
                        help="folder with the output files of the i-PI normal analysis (default: 'vib')", default="vib")
    parser.add_argument("-o",  "--output",        type=str, **argv, 
                        help="output file (default: 'normal-modes.pickle')", default="normal-modes.pickle")
    return parser.parse_args()
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

    #---------------------------------------#
    # read reference atomic structure
    reference = None
    if args.reference is not None:
        print("\tReading reference atomic structure from input '{:s}' ... ".format(args.reference), end="")
        reference = read(args.reference,index=0)
        print("done")

    #---------------------------------------#
    # phonon modes
    print("\n\tReading normal modes from folder '{:s}' ... ".format(args.folder),end="")
    # pm = pd.DataFrame(index=[gamma],columns=["q","freq","modes"])
    nm = NormalModes.from_folder(folder=args.folder,ref=reference)
    # pm.at[gamma,"q"]     = gamma
    # pm.at[gamma,"modes"] = nm 
    # if reference is not None:
    #     nm.set_reference(reference)
    print("done")

    #---------------------------------------#
    print("\n\tWriting normal modes to file '{:s}' ... ".format(args.output), end="")
    nm.to_pickle(args.output)
    print("done")
    
    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
