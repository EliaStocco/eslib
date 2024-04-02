#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, warning, float_format
from eslib.tools import convert
import numpy as np
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Convert the BECs in a extxyz file to a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="atomic structures file [extxyz]")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-k" , "--keyword"     , **argv, required=False, type=str, help="keyword for BEC tensors (default: 'bec')", default="bec")
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output file with the BEC tensors (default: 'bec.txt')", default='bec.txt')
    return parser# .parse_args()
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory:List[Atoms] = AtomicStructures.from_file(file=args.input,format=args.input_format,index=":")
    print("done")

    #------------------#
    # extract
    N = len(trajectory)
    all_bec = {
        "x":[None]*N,
        "y":[None]*N,
        "z":[None]*N,
    }
    for key in all_bec.keys():
        name = "{:s}{:s}".format(args.keyword,key)
        print("\tExtracting '{:s}' from the trajectory ... ".format(name), end="")
        for n,atoms in enumerate(trajectory):
            all_bec[key][n] = atoms.arrays[name]
        print("done")

    #------------------#
    # reshape
    BEC = [None]*N
    for n in range(N):
        x = np.asarray(all_bec["x"][n]).flatten()
        y = np.asarray(all_bec["y"][n]).flatten()
        z = np.asarray(all_bec["z"][n]).flatten()
        BEC[n] = np.column_stack((x,y,z))
    
    #------------------#
    print("\n\tWriting the BEC tensors to file '{:s}' ... ".format(args.output), end="")
    #try:
    with open(args.output,"w") as file:
        for n,bec in enumerate(BEC):
            np.savetxt(file,bec,fmt=float_format,header="structure {:d}".format(n))
    print("done")
    # except Exception as e:
    #     print("\n\tError: {:s}".format(e))
    
#---------------------------------------#
if __name__ == "__main__":
    main()