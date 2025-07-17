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
    parser.add_argument("-a" , "--atom"          , **argv, required=True , type=int  , help="atom index (0-based) to be kept fixed")
    parser.add_argument("-p" , "--position"      , **argv, required=True , type=flist, help="positions of the atoms to be fixed [angstroms]")
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
    print("\n\tComputing the displacements ... ", end="")
    atom_position = np.zeros((len(structure),3))
    for i in range(len(structure)):
        atom_position[i,:] = structure[i].get_positions()[args.atom,:]
    displacements = atom_position - np.array(args.position).reshape((1,3))
    print("done")
    
    #------------------#
    print("\n\tModifying positions ... ", end="")
    for i in range(len(structure)):
        pos = structure[i].get_positions()
        pos[:,:] -= displacements[i,:]
        structure[i].set_positions(pos)
    print("done")
    
    #------------------#
    print("\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    structure.to_file(file=args.output,format=args.output_format)
    print("done")
    
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
