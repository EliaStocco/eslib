#!/usr/bin/env python
import argparse
import numpy as np
from ase.io import write
from eslib.classes.trajectory import AtomicStructures, info, array


#---------------------------------------#
# Description of the script's purpose
description = "Save an 'array' or 'info' from an extxyz file to a txt file."
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
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv,type=str, help="input file [extxyz]")
    parser.add_argument("-n" , "--name"  , **argv,type=str, help="name for the new info/array")
    parser.add_argument("-w" , "--what"  , **argv,type=str, help="what the data is: 'i' (info) or 'a' (arrays)")
    parser.add_argument("-o" , "--output", **argv,type=str, help="output file (default: '[name].txt')", default=None)
    parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: '%%24.18f')", default='%24.18f')
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

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input)
    print("done")

    #---------------------------------------#
    # reshape
    print("\tExtracting '{:s}' from the trajectory ... ".format(args.name), end="")
    N = len(atoms)
    Natoms = atoms[0].positions.shape[0]
    if args.what in ['a','arrays','array']:
        data = array(atoms,args.name)
    elif args.what in ['i','info']:
        data = info(atoms,args.name)  
        what = "info"
    else:
        raise ValueError("'what' (-w,--what) can be only 'i' (info), or 'a' (array)")
    print("done")

    print("\t'{:s}' shape: ".format(args.name),data.shape)

    #---------------------------------------#
    # store
    if args.output is None:
        file = "{:s}.txt".format(args.name)
    else:
        file = args.output
    print("\tStoring '{:s}' to file '{:s}' ... ".format(args.name,file), end="")
    np.savetxt(file,data,fmt=args.output_format) # fmt)
    print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()

