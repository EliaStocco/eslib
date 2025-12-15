#!/usr/bin/env python
import numpy as np
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format

#---------------------------------------#
# Description of the script's purpose
description = "Center the atomic structures along the z direction."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-z" , "--z_cellpar"     , **argv, required=False, type=float, help="modify z cellpar (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of", "--output_format" , **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    if args.z_cellpar is not None:
        print("\tModifying cellpar ... ", end="")
        for n,structure in enumerate(structures):
            cellpar = structure.cell.cellpar()
            cellpar[2] = args.z_cellpar
            structure.cell = Cell.fromcellpar(cellpar)
        print("done")
        
    #------------------#
    print("\tComputing center of mass ... ", end="")
    com = np.zeros((len(structures),3))
    for n,structure in enumerate(structures):
        com[n] = structure.get_center_of_mass()
        com[n][2] = structure.cell.cellpar()[2]/2.
    print("done")
    
    #------------------#
    print("\tSetting the center of mass ... ", end="")
    for n,structure in enumerate(structures):
        structure.set_center_of_mass(com[n])
    print("done")
    
    #------------------#
    print(f"\tWriting atomic structure to '{args.output}' ... ", end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")
        
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
