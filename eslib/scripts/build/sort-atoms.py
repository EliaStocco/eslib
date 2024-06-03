#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from eslib.tools import sort_atoms


#---------------------------------------#
# Description of the script's purpose
description = "Sort atoms of an atomic structure depending on the interatomic distances w.r.t. another structure."
closure = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#

# colors
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description = Fore.GREEN + Style.BRIGHT + description + Style.RESET_ALL
    closure = Fore.BLUE + Style.BRIGHT + closure + Style.RESET_ALL
    input_arguments = Fore.GREEN + Style.NORMAL + input_arguments + Style.RESET_ALL
except:
    pass

def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b", }
    parser.add_argument("-r", "--reference", type=str, **argv, help="reference atomic structure")
    parser.add_argument("-s", "--structure", type=str, **argv, help="atomic structure to be sorted")
    parser.add_argument("-f", "--format"   , type=str, **argv, help="file format for both the input and the output (default: %(default)s)" , default=None)
    parser.add_argument("-i", "--indices"  , type=str, **argv, help="output txt with the indices used to sort the structure (default: %(default)s)", default=None)
    parser.add_argument("-o", "--output"   , type=str, **argv, help="output file for the sorted structure (default: %(default)s)", default="sorted.xyz")
    return parser# .parse_args()

#---------------------------------------#
def main():

    #-------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k), getattr(args, k))
    print()

    #-------------------#
    # reference structure
    print("\tReading reference atomic structures from file '{:s}' ... ".format(args.reference), end="")
    reference = read(args.reference, format=args.format, index=0)
    print("done")

    #-------------------#
    # structure to be sorted
    print("\tReading atomic structures to be sorted from file '{:s}' ... ".format(args.structure), end="")
    structure = read(args.structure, format=args.format, index=0)
    print("done")

    #-------------------#
    # sort
    print("\n\tSorting the atoms  ... ", end="")
    sorted, indices = sort_atoms(reference, structure)
    print("done")

    #-------------------#
    if args.indices is not None:
        print("\n\tWriting the indices used to sort to file '{:s}'  ... ".format(args.indices), end="")
        np.savetxt(args.indices,indices)
        print("done")

    #-------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting sorted atomic structure to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, sorted, format=args.format) # fmt)
        print("done")
    except Exception as e:
        print(f"\n\tError: {e}")
    
    #-------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
