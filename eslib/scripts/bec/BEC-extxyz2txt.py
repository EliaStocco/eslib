#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning, float_format
from eslib.tools import convert
from eslib.input import str2bool
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
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"     , **argv, required=False, type=str, help="keyword for BEC tensors (default: %(default)s)", default="bec")
    parser.add_argument("-c" , "--check"     , **argv, required=False, type=str2bool, help="check that BEC tensors are correctly formatted (default: %(default)s)", default=True)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output file with the BEC tensors (default: %(default)s)", default='bec.txt')
    return parser# .parse_args()
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input,format=args.input_format,index=":")
    print("done\n")

    #------------------#
    # extract
    N = len(trajectory)
    Natoms = trajectory[0].get_global_number_of_atoms()
    shape = (N,Natoms)
    all_bec = {
        "x":np.full(shape,np.nan),
        "y":np.full(shape,np.nan),
        "z":np.full(shape,np.nan),
    }
    for key in all_bec.keys():
        name = "{:s}{:s}".format(args.keyword,key)
        print("\tExtracting '{:s}' from the trajectory".format(name), end="")
        all_bec[key] = trajectory.get(name)
        print(" --> shape: ",all_bec[key].shape," ... ",end="")
        print("done")
    
    shape = (len(trajectory),3*Natoms,3)
    BEC = np.full(shape,np.nan)
    print("\n\tBEC.shape: ",BEC.shape)

    #------------------#
    for n,key in enumerate(["x","y","z"]):
        tmp = all_bec[key]
        BEC[:,:,n] = tmp.reshape((N,3*Natoms))

    assert np.isnan(BEC).sum() == 0, "Found nan values in BEC"

    if args.check:
        if not trajectory.is_there(args.keyword):
            print("\n\t{:s}: '{:s}' not found --> it's not possible to check whether BECs are correctly formatted.".format(warning,args.keyword))
        else:
            tmp = trajectory.get(args.keyword)
            tmp = tmp.reshape((tmp.shape[0],-1,3))
            if not np.allclose(BEC,tmp):
                print("\t{:s}: '{:s}' and the ones reconstructued differ".format(warning,args.keyword))
    
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