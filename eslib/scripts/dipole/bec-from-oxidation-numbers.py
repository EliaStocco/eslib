#!/usr/bin/env python
from ase.io import write
import numpy as np
import json
from ase import Atoms
from eslib.classes.trajectory import trajectory as Trajectory
from eslib.formatting import esfmt, warning, float_format
from eslib.physics import bec_from_oxidation_number

#---------------------------------------#
# Description of the script's purpose
description = "Compute the BECs from the nominal oxidation numbers."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="extxyz input file with an atomic structure and oxidation numbers")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-k" , "--keyword"       , **argv, required=False, type=str, help="keyword for the oxidation numbers (default: 'oxidation-numbers')", default='oxidation-numbers')
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="txt output file with the BECs (default: 'bec.txt')", default='bec.txt')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = Trajectory(args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    print("\tExtracting oxidation numbers from '{:s}' ... ".format(args.keyword), end="")
    on = atoms.arrays[args.keyword]
    print("done")

    #------------------#
    print("\tCreating BECs ... ", end="")
    bec = bec_from_oxidation_number(atoms,on)
    print("done")

    #------------------#
    
    if not bec.check_asr(0):
        print("\t{:s}: the BECs do not satisfy the acoutsitc sum rule --> subtracting the average.".format(warning))
        mean = bec.force_asr()
        print("\tSubtracted average: {:f}".format(mean.isel(structure=0)))
        if not bec.check_asr(0):
            raise ValueError("coding error")

    #------------------#    
    args.output = str(args.output)
    print("\n\tSaving BECs to file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output, bec.isel(structure=0).to_numpy(),fmt=float_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()