#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import str2bool
from ase.spacegroup import get_spacegroup, Spacegroup
from spglib import standardize_cell
from ase import Atoms
from ase.cell import Cell

# Documentation:
# - https://github.com/ajjackson/ase-tutorial-symmetry/blob/master/ase-symmetry.md

#---------------------------------------#
description = "Find the symmetry of an atomic structure."
    
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , type=str     , **argv, required=True , help="atomic structure input file")
    parser.add_argument("-if", "--input_format" , type=str     , **argv, required=False, help="input file format (default: %(default)s)" , default=None)
    # parser.add_argument("-r" , "--rattle"       , type=float, **argv, required=False, help="threshold to find symmetry (default: %(default)s)" , default=0.0)
    parser.add_argument("-t" , "--threshold"    , type=float   , **argv, required=False, help="threshold to find symmetry (default: %(default)s)" , default=1e-3)
    parser.add_argument("-p" , "--primitive"    , type=str2bool, **argv, required=False, help="to primitive (default: %(default)s)" , default=True)
    parser.add_argument("-r" , "--rotate"       , type=str2bool, **argv, required=False, help="rotate cell (default: %(default)s)" , default=True)
    parser.add_argument("-o" , "--output"       , type=str     , **argv, required=True , help="output file")
    parser.add_argument("-of", "--output_format", type=str     , **argv, required=False, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    structure = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    # Get the initial space group of the structure
    spacegroup:Spacegroup = get_spacegroup(structure,symprec=args.threshold)
    print(f'\tInitial space group: {spacegroup.no, spacegroup.symbol}')

    # #------------------#
    # if args.rattle > 0:
    #     print("\n\tRattling structure ... ", end="")
    #     structure.rattle(stdev=args.rattle)
    #     print("done")

    #     #------------------#
    #     spacegroup:Spacegroup = get_spacegroup(structure,symprec=args.threshold)
    #     print(f'\tRattled space group: {spacegroup.no, spacegroup.symbol}')
    

    #------------------#
    # Refine the structure
    print("\n\tRefining symmetry ... ", end="")
    cell = (structure.cell, structure.get_scaled_positions(), structure.numbers)
    refined_structure = standardize_cell(cell, to_primitive=args.primitive, symprec=args.threshold)
    new_unit_cell, new_scaled_positions, new_numbers = refined_structure
    refined_structure = Atoms(new_numbers, cell=new_unit_cell, scaled_positions=new_scaled_positions)
    print("done")

    #------------------#
    # Refine the structure
    spacegroup:Spacegroup = get_spacegroup(refined_structure,symprec=args.threshold)
    print(f'\tRefined space group: {spacegroup.no, spacegroup.symbol}')

    if args.rotate:
        print("\n\tRotating cell ... ", end="")
        cellpar = refined_structure.cell.cellpar()
        cell = Cell.fromcellpar(cellpar).array
        refined_structure.set_cell(cell,scale_atoms=True)
        print("done")

    #------------------#
    # Save the refined structure to a file
    print("\n\tWriting the refined atomic structure to file '{:s}' ... ".format(args.output), end="")
    AtomicStructures([refined_structure]).to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()