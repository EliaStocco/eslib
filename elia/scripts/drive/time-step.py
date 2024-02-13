#!/usr/bin/env python
import numpy as np
from elia.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Compute a good time step for the dynamics given the angular frequency of the driving pulse (or phonon mode)."
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
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-w" , "--frequency"   , type=float   , **argv, help="angular frequency")
    parser.add_argument("-u" , "--unit"        , type=str     , **argv, help="unit of the angular frequency (default: 'THz')", default="thz")
    parser.add_argument("-n" , "--number"      , type=int     , **argv, help="number of integration steps within one period (default: 10)", default=10)
    parser.add_argument("-o" , "--output_unit" , type=str     , **argv, help="output unit for the time step (default: 'femtosecond')", default="femtosecond")
    return parser.parse_args()

#---------------------------------------#
def main():

    #---------------------------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #------------------#
    w = convert(args.frequency,"frequency",args.unit,"atomic_unit")
    Tau = 2*np.pi/w
    Tfs = convert(Tau,"time","atomic_unit",args.output_unit)
    dt = Tfs/args.number

    print("\n\t   period [{:s}]: {:.2f}".format(args.output_unit,Tfs))
    print(  "\ttime step [{:s}]: {:.2f}".format(args.output_unit,dt))

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()



print()