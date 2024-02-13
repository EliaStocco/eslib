#!/usr/bin/env python
import argparse
import numpy as np
from ase.io import read
from copy import copy
from eslib.tools import cart2lattice
from eslib.input import size_type

description = "Compute the best dipole quantum to be shifted to a dataset.\n"
message = "\t!Attention:\n"+\
    "\t- you need to provide the data as a 'extxyz' file\n"+\
    "\t- the atomic structure have to represent an MD trajectory (the order matters!)"+\
    "\n"
DEBUG=False

def prepare_args():

    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        ,   **argv,type=str     , 
                        help="input file")
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str     , 
                        help="input file format (default: 'None')" , default=None)
    parser.add_argument("-s", "--shift",  type=lambda s: size_type(s,dtype=float,N=3), default=None, **argv,
                        help="additional arrays to be added to the output file (default: [0,0,0])")
    
    return parser.parse_args()

def main():

    ###
    # Parse the command-line arguments
    args = prepare_args()

    ###
    # Print the script's description
    print("\n\t{:s}".format(description))

    if args.shift is not None:
        if len(args.shift) != 3 :
            raise ValueError("You should provide 3 integer numbers as shift vectors")
    
    print(message)

    ###
    # read the MD trajectory from file
    print("\tReading (the first only) atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format)
    print("done")

    ###
    # convert
    cell = np.asarray(atoms.cell.array).T
    R = cart2lattice(cell)
    lenght = np.linalg.norm(cell,axis=0)
    shift = R @ args.shift / lenght
    print("\tConverted the shift from cartesian to lattice coordinates: ",np.round(shift,0))
    print("\tShift with all digits: ",shift)
    
    # Script completion message
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()
