#!/usr/bin/env python
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist

#---------------------------------------#
# Description of the script's purpose
description = "Cut a 2D structure."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-x", "--x_lims"      , **argv, required=True , type=flist, help="x limits")
    parser.add_argument("-y", "--y_lims"      , **argv, required=True , type=flist, help="y limits")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    print("\tCutting structures ... ",end="")
    for n,atoms in enumerate(structures):
        positions = atoms.positions
        x = positions[:, 0]
        y = positions[:, 1]
        mask_x = (x >= args.x_lims[0]) & (x <= args.x_lims[1])
        mask_y = (y >= args.y_lims[0]) & (y <= args.y_lims[1])
        mask = mask_x & mask_y
        structures[n] = atoms[mask]
    print("done")
    
    print("\n\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

    

#---------------------------------------#
if __name__ == "__main__":
    main()