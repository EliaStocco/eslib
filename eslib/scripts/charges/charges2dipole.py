#!/usr/bin/env python
import json
# from typing import List

import numpy as np
# from ase import Atoms

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models.dipole import DipolePartialCharges
from eslib.formatting import esfmt, warning
from eslib.show import show_dict
from eslib.tools import is_integer

#---------------------------------------#
# Description of the script's purpose
description = "Compute the point-charges dipole."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-d" , "--dipole"        , **argv, required=False, type=str, help="name of the dipole (default: %(default)s)", default='dipole')
    parser.add_argument("-c" , "--charges"       , **argv, required=True , type=str, help="name of the charges (default: %(default)s)", default='Qs')
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file with the atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    print("\n\tExtracting charges ... ",end="")
    charges = structures.get(args.charges)
    print("done")
    print("\tcharges.shape: ",charges.shape)
    
    print("\tExtracting positions ... ",end="")
    positions = structures.get("positions")
    print("done")
    print("\tpositions.shape: ",positions.shape)

    #------------------#
    print("\n\tComputing dipoles ... ",end="")
    dipole = np.einsum("ijk,ij->ik",positions,charges)
    print("done")
    print("\tdipole.shape: ",dipole.shape)
    
    print(f"\n\tSetting dipoles as '{args.dipole}' ... ",end="")
    structures.set(args.dipole,dipole,"info")
    print("done")
    
    #------------------#
    print("\n\tWriting the atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()