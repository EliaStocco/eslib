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
    parser.add_argument("-c" , "--com"           , **argv, required=True , type=flist, help="file with an atomic structure")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    #------------------#
    print("\n\tSetting the center of mass ... ", end="")
    structure.set_center_of_mass(args.com)
    print("done")
    
    #------------------#
    print("\tCenter of mass [ang]:",structure.get_center_of_mass())
    
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
