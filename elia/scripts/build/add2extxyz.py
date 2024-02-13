#!/usr/bin/env python
import numpy as np
from ase.io import write
from elia.classes.trajectory import trajectory
from elia.formatting import warning, error, esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Add data to an extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv,type=str, help="input file [extxyz]")
    parser.add_argument("-n" , "--name"  , **argv,type=str, help="name for the new info/array")
    parser.add_argument("-d" , "--data"  , **argv,type=str, help="file (txt or csv) with the data to add")
    parser.add_argument("-w" , "--what"  , **argv,type=str, help="what the data is: 'i' (info) or 'a' (arrays)")
    parser.add_argument("-o" , "--output", **argv,type=str, help="output file (default: 'output.extxyz')", default="output.extxyz")
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = trajectory(args.input)
    print("done")
    N = len(atoms) 
    print("\tn. of atomic structures: ",N)

    #---------------------------------------#
    # data
    print("\tReading data from file '{:s}' ... ".format(args.data), end="")
    data = np.loadtxt(args.data)
    print("done")

    print("\tData shape: ",data.shape)

    #---------------------------------------#
    if data.shape[0] != N:
        print("\t{:s}: shapes do not match --> keeping only the last {:d} elements of the array to be stored".format(warning,N))
        data = data[-N:]
    #---------------------------------------#
    # reshape
    Natoms = atoms[0].positions.shape[0]
    if args.what in ['a','arrays','array']:
        data = data.reshape((N,Natoms,-1))
        what = "arrays"
    elif args.what in ['i','info']:
        data = data.reshape((N,-1))    
        what = "info"
    else:
        raise ValueError("'what' (-w,--what) can be only 'i' (info), or 'a' (array)")
    print("\tData reshaped to: ",data.shape)

    #---------------------------------------#
    # store
    print("\tStoring data to '{:s}' with name '{}' ... ".format(what,args.name), end="")
    atoms = list(atoms)
    for n in range(N):
        if what == "info":
            atoms[n].info[args.name] = data[n]
        elif what == "arrays":
            atoms[n].arrays[args.name] = data[n]
        else:
            raise ValueError("internal error")

    #---------------------------------------#
    print("\n\tWriting trajectory to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, list(atoms)) # fmt)
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

if __name__ == "__main__":
    main()

