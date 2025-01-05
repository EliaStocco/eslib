#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format

#---------------------------------------#
# Description of the script's purpose
description = "Compute the center of mass of some structures."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    print("\n\tComputing center of mass ... ", end="")
    com = np.zeros((len(structures),3))
    for n,structure in enumerate(structures):
        com[n] = structure.get_center_of_mass()
    print("done")
    
    #------------------#
    if args.output is not None:
        print("\tWriting center of mass to file '{:s}' ... ".format(args.output), end="")
        np.savetxt(args.output,com,fmt=float_format)
        print("done")
    
    #------------------#    
    print("\tCenter of mass [ang]:",com)
        
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
