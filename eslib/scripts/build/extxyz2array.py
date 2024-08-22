#!/usr/bin/env python
import numpy as np
from ase.io import write
from eslib.classes.unify_traj import Trajectory
from eslib.classes.trajectory import AtomicStructures
from eslib.classes.properties import Properties
from eslib.formatting import esfmt, float_format
from eslib.input import ilist
from eslib.classes.physical_tensor import PhysicalTensor

#---------------------------------------#
# Description of the script's purpose
description = "Save an 'array' or 'info' from an extxyz file to a txt file."


#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv,type=str, help="input file [extxyz]")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--name"  , **argv,type=str, help="name for the new info/array")
    parser.add_argument("-s" , "--shape"  , **argv,type=ilist, help="reshape the data (default: %(default)s", default=None)  
    parser.add_argument("-o" , "--output", **argv,type=str, help="output file (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: %(default)s)", default=float_format)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    file = str(args.input)
    if file.endswith("pickle"):
        atoms = Trajectory.from_file(file=file,format=args.input_format)
    elif file.endswith("txt") or file.endswith("out"):
        atoms = AtomicStructures.from_file(file=file,format=args.input_format)
    else:
        atoms = AtomicStructures.from_file(file=file,format=args.input_format)
    print("done")

    #---------------------------------------#
    # reshape
    print("\tExtracting '{:s}' from the trajectory ... ".format(args.name), end="")
    data = atoms.get(args.name)  
    print("done")
    print("\t'{:s}' shape: ".format(args.name),data.shape)

    print("\n\tData type: ",data.dtype)

    if np.issubdtype(data.dtype, np.str_):
        print("\tData contains strings: -of/--output_format wil be set to '%s'" % "%s")
        args.output_format = "%s"

    if args.shape is not None:
        shape = tuple(args.shape)
        print("\tReshaping data to ",shape," ... ",end="")
        data = data.reshape(shape)
        print("done")

    #---------------------------------------#
    print("\n\tConverting data into PhysicalTensor ... ", end="")
    data = PhysicalTensor(data)
    print("done")

    #---------------------------------------#
    # store
    if args.output is None:
        file = "{:s}.txt".format(args.name)
    else:
        file = str(args.output)

    print("\tStoring '{:s}' to file '{:s}' ... ".format(args.name,file), end="")
    data.to_file(file=file,fmt=args.output_format)
    # if file.endswith("txt"):
    #     np.savetxt(file,data,fmt=args.output_format) # fmt)
    # elif file.endswith("npy"):
    #     np.save(file,data)
    # else:
    #     raise ValueError("Only `txt` and `npy` files are supported.")
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()

