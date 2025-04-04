#!/usr/bin/env python
from typing import List

from ase import Atoms
from ase.io import read, write

from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Simple file conversion only based on ASE."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file with the oxidation numbers (default: %(default)s)", default="oxidation-numbers.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:List[Atoms] = read(filename=args.input,format=args.input_format,index=":")
    print("done")
    
       
    print("\n\tWriting the atomic structures to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=atoms,filename=args.output,format=args.output_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    

#---------------------------------------#
if __name__ == "__main__":
    main()