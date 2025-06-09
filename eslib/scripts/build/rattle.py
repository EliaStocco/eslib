#!/usr/bin/env python
from ase import Atoms
import numpy as np
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Rattle structures."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-s"  , "--stdev"       , **argv, required=True , type=float, help="standard deviation of the normal distribution used for rattling")
    parser.add_argument("-o"  , "--output"       , **argv, required=True , type=str  , help="output file with the atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # cell
    print("\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #------------------#
    def rattle(atoms:Atoms):
        atoms.rattle(stdev=args.stdev)
    
    print("\tRattling structures ... ", end="")
    structures.apply(func=rattle,parallel=False)
    print("done")
    
    #------------------#
    print("\n\tSaving the atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
    

#---------------------------------------#
if __name__ == "__main__":
    main()