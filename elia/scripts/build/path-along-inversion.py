#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from elia.tools import distance
from elia.tools import segment
from elia.input import size_type, str2bool


#---------------------------------------#
# Description of the script's purpose
description = "Create a path along a path where the positions get inverted (R --> -R) along the specified crystal/cartesian axes."
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
    parser.add_argument("-i" , "--input"  , type=str                             , **argv, help="input atomic structure")
    parser.add_argument("-f" , "--format" , type=str                             , **argv, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-a" , "--axes"   , type=lambda s: size_type(s,dtype=str), **argv, help="axes w.r.t. invert the positions [1,2,3,x,y,z] (default: [1,2,3])" , default=["1","2","3"])
    parser.add_argument("-s" , "--sort"   , type=str2bool                        , **argv, help="whether to sort the second structure (dafault: true)", default=True)
    parser.add_argument("-n", "--number"  , type=int                             , **argv, help="number of inner structures (default: 10)", default=10)
    #parser.add_argument("-oo", "--output_original" , type=str                    , **argv, help="output file for the original structure with wrapper atoms (default: 'original.extxyz')", default="original.xyz")
    parser.add_argument("-o" , "--output" , type=str                             , **argv, help="output extxyz file for the path (default: 'path.extxyz')", default="path.extxyz")
    return parser.parse_args()

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
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    structure = read(args.input, format=args.format, index=0)
    print("done")

    #-------------------#
    # original
    structure.wrap()
    original = structure.copy()

    # #-------------------#
    # print("\n\tWriting original atomic structure to file '{:s}' ... ".format(args.output_original), end="")
    # try:
    #     write(args.output_original, structure, format=args.format)
    #     print("done")
    # except Exception as e:
    #     print(f"\n\tError: {e}")


    #-------------------#
    # structure to be sorted
    xyz2num = {"x":0,"y":1,"z":2}
    print("\n\tInverting positions along specified axes:")
    for axis in args.axes:
        print("\t\talong {:s} axis ... ".format(axis),end="")
        if axis in ["1","2","3"]:
            positions = structure.get_positions()
            axis = int(axis)-1
            positions[:,axis] = -positions[:,axis]
            structure.set_positions(positions)
        elif axis in ["x","y","z"]:
            positions = structure.get_scaled_positions()
            axis = xyz2num[axis]
            positions[:,axis] = -positions[:,axis]
            structure.set_scaled_positions(positions)
        else:
            raise ValueError("axis not known")
        print("done")

    # #-------------------#
    # # fold within primitive cell
    # print("\n\tFolding atoms within the primitive cell  ... ", end="")
    # structure.wrap()
    # print("done")

    #-------------------#
    # sort
    print("\n\tSorting the atoms of the second structure  ... ", end="")
    _, A, B = distance(original, structure)
    print("done")

    #-------------------#
    print("\n\tComputing the path positions ... ", end="")
    pathpos = segment(A.positions,B.positions,N=args.number)
    print("done")

    #-------------------#
    N = pathpos.shape[0]
    print("\tn. of structures in the path: '{:d}'".format(N))

    #-------------------#
    print("\tCreating the path ... ", end="")
    path = [None]*N
    for n in range(N):
        path[n] = A.copy()
        path[n].set_positions(pathpos[n])
    print("done")

    #-------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting inverted atomic structure to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, path, format="extxyz") # fmt)
        print("done")
    except Exception as e:
        print(f"\n\tError: {e}")
    
    #-------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
