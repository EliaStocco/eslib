#!/usr/bin/env python
from elia.classes.bec import bec as BEC
from elia.classes.trajectory import array
from elia.classes.trajectory import trajectory as Trajectory 

#---------------------------------------#
# Description of the script's purpose
description = "Template for a script."
error = "***Error***"
closure = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"        , **argv, type=str, help="input extxyz file")
    parser.add_argument("-if" , "--input_format" , **argv, type=str, help="input file format (default: 'None')", default=None)
    parser.add_argument("-n"  , "--name"         , **argv, type=str, help="name for the array where BEC are stores (default: 'bec')", default="bec")
    parser.add_argument("-o"  , "--output"       , **argv, type=str, help="output *.pickle file")
    return parser.parse_args()

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
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    trajectory = Trajectory(args.input, format=args.input_format)  #eV
    print("done")

    #------------------#
    print("\tExtracting BECs from '{:s}' array of the trajectory ... ".format(args.name),end="")
    becarr = array(trajectory,args.name)
    print("done")

    print("\tBECs extracted as array with shape: ",becarr.shape)

    #------------------#
    print("\n\tRecasting BECs as a python object ... ",end="")
    bec = BEC.from_numpy(becarr)
    print("done")

    #------------------#
    print("\n\tSaving BECs object to file '{:s}' ... ".format(args.output),end="")
    bec.to_pickle(args.output)
    print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

    return 

#---------------------------------------#
if __name__ == "__main__":
    main()
