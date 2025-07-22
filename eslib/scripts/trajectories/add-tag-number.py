#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.io_tools import extract_number_from_filename, pattern2sorted_files

#---------------------------------------#
# Description of the script's purpose
description = "Add tags (info) to an extxyz file based on the number found in the file path."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"      , **argv, type=str, required=True , help="keyword")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    files = pattern2sorted_files(args.input)
    tags = [ int(extract_number_from_filename(x)) for x in files ]
    tags = np.sort(tags)
    print(f"\tFound {len(files)} files:")
    for n,f in zip(tags,files):
        print(f"\t - {n}: {f}")

    #------------------#
    # atomic structures
    print("\n\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)
    
    #------------------#
    print("\n\tStoring tags ... ", end="")
    structures.set(args.keyword,tags,"info")
    print("done")
    
    #------------------#
    print("\n\tWriting trajectory to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

if __name__ == "__main__":
    main()

