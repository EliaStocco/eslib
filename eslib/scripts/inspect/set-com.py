#!/usr/bin/env python
import numpy as np
from ase import Atoms

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist

#---------------------------------------#
# Description of the script's purpose
description = "Set the center of mass of a structure."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-c" , "--com"           , **argv, required=True , type=flist, help="center of mass")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structure = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    print("\n\tSetting the center of mass ... ", end="")
    for i in range(len(structure)):
        structure[i].set_center_of_mass(args.com)
    print("done")
    
    #------------------#
    print("\tCenter of mass [ang]:",structure[0].get_center_of_mass())
    
    #------------------#
    print("\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    structure.to_file(file=args.output,format=args.output_format)
    print("done")
    
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
