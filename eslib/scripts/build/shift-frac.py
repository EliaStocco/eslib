#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from ase import Atoms
from eslib.formatting import esfmt
from eslib.input import flist


#---------------------------------------#
# Description of the script's purpose
description = "Change the fractional coordinates of one atom."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n"  , "--atom_index"   , **argv, required=True , type=int  , help="atom inde to shift")
    parser.add_argument("-s"  , "--shift"       , **argv, required=False, type=flist, help="list with the shift vector" , default=[0,0,0])
    parser.add_argument("-o"  , "--output"       , **argv, required=True , type=str  , help="output file with the atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print(f"\tReading the atomic structures from file '{args.input}' ... ", end="")
    atoms:Atoms = read(filename=args.input,format=args.input_format,index=0)
    print("done")
    
    #------------------#
    print("\tShifting coordinates ... ", end="")
    frac_pos = atoms.get_scaled_positions()
    frac_pos[args.atom_index] += np.asarray(args.shift)
    atoms.set_scaled_positions(frac_pos)
    print("done")
    
    #------------------#
    print(f"\tSaving the atomic structure to file '{args.output}' ... ", end="")
    write(filename=args.output,images=atoms,format=args.output_format)
    print("done")
    

#---------------------------------------#
if __name__ == "__main__":
    main()
