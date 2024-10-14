#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Add the lattice vectors to an atomic structures."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with the atomic structures to be modified")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-c" , "--cell"          , **argv, required=True , type=flist, help="a, b, c, α, β, γ [ang,deg]")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file with the modified atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    #------------------#
    print("\tConstructing the cell ... ", end="")
    cell = Cell.fromcellpar(args.cell)
    print("done")
    
    #------------------#
    # replace
    print("\tReplacing cell ... ", end="")
    atoms.set_cell(cell)
    print("done")

    #------------------#
    print("\n\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    write(images=atoms,filename=args.output, format=args.output_format) # fmt)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()