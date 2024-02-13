#!/usr/bin/env python
from ase.io import read, write
import argparse
from elia.input import slist
from elia.formatting import esfmt, error

#---------------------------------------#
# Description of the script's purpose
description = "Fold the atomic structures into the primitive cell."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        , **argv, type=slist, help="list of input files (example [fileA.xyz,fileB.cif])")
    parser.add_argument("-o"  , "--output"       , **argv, type=str  , help="output file")
    parser.add_argument("-of" , "--output_format", **argv, type=str  , help="output file format (default: 'None')", default=None)
    options = parser.parse_args()
    return options

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #------------------#
    trajectory = [None]*len(args.input)

    #------------------#
    for n,file in enumerate(args.input):
        print("\tReading atomic structures from input file '{:s}' ... ".format(file), end="")
        trajectory[n] = read(file,index=":")
        print("done")
    
    #------------------#
    print("\tAdding information 'original-file' to each trajectory:")
    for n,file in enumerate(args.input):
        print("\t{:2d}: {:s}".format(n,file))
        for i in range(len(trajectory)):
            trajectory[n][i].info["original-file"] = n

    #------------------#
    print("\tConcatenating all the trajectories ... ", end="")
    single_trajectory = [item for sublist in trajectory for item in sublist]
    print("done")

    #------------------#
    print("\n\tWriting concatenated structures to output file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, single_trajectory, format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

    return

#---------------------------------------#
if __name__ == "__main__":
    main()