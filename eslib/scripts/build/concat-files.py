#!/usr/bin/env python
from ase.io import read, write

import glob
from eslib.input import slist
from eslib.formatting import esfmt, error
from eslib.classes.trajectory import AtomicStructures
from ase import Atoms
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Fold the atomic structures into the primitive cell."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        , **argv, type=slist, help="list of input files (example [fileA.xyz,fileB.cif])")
    parser.add_argument("-o"  , "--output"       , **argv, type=str  , help="output file")
    parser.add_argument("-of" , "--output_format", **argv, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #------------------#
    matched_files = glob.glob(args.input[0])
    if matched_files is None or len(matched_files):
        args.input = matched_files
    
    #------------------#
    print("\tReading atomic structures from file:")
    trajectory:List[List[Atoms]] = [None]*len(args.input)
    for n,file in enumerate(args.input):
        print("\t\t{:d}: '{:s}' ... ".format(n,file), end="")
        trajectory[n] = read(file,index=":")
        print("done --> (n. atomic structures: {:d})".format(len(trajectory[n])))
    
    #------------------#
    print("\n\tAdding information 'original-file' to each trajectory:")
    for n,file in enumerate(args.input):
        print("\t\t{:2d}: {:s}".format(n,file))
        for i in range(len(trajectory[n])):
            trajectory[n][i].info["original-file"] = n

    #------------------#
    print("\n\tConcatenating all the trajectories ... ", end="")
    trajectory = AtomicStructures([item for sublist in trajectory for item in sublist])
    print("done")
    print("\tn. of atomic structures: ",len(trajectory))

    #------------------#
    print("\n\tWriting concatenated structures to output file '{:s}' ... ".format(args.output), end="")
    try:
        trajectory.to_file(file=args.output, format=args.output_format)
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

    return

#---------------------------------------#
if __name__ == "__main__":
    main()