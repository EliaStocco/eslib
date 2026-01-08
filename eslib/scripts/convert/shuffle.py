#!/usr/bin/env python
import numpy as np 
from eslib.formatting import esfmt
from eslib.classes.atomic_structures import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Shuffle a trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"            , **argv,required=True , type=str     , help="input file")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o"  , "--output"           , **argv,required=True , type=str     , help="output file")
    parser.add_argument("-of" , "--output_format"    , **argv,required=False, type=str     , help="output file format (default: %(default)s)", default=None)
    return parser


#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    print("\tShuffling ... ", end="")
    np.random.seed(0)
    ii = np.arange(len(structures))
    np.random.shuffle(ii)
    structures = structures.subsample(indices=ii)
    print("done")
    
    #-------------------#
    print("\n\tWriting structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()