#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import warning, esfmt
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Add data to an extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"    , **argv,type=str     , required=True , help="input file [extxyz]")
    parser.add_argument("-n" , "--name"     , **argv,type=str     , required=True , help="name for the new info/array")
    parser.add_argument("-d" , "--data"     , **argv,type=str     , required=True , help="file (txt or csv) with the data to add")
    parser.add_argument("-w" , "--what"     , **argv,type=str     , required=True , help="what the data is: 'i' (info) or 'a' (arrays)")
    parser.add_argument("-r" , "--replicate", **argv,type=str2bool, required=False, help="replicate the same array for each structure", default=False)
    parser.add_argument("-o" , "--output"   , **argv,type=str     , required=True , help="output file (default: %(default)s)", default="output.extxyz")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input)
    print("done")
    N = len(atoms) 
    print("\tn. of atomic structures: ",N)

    #---------------------------------------#
    # data
    print("\tReading data from file '{:s}' ... ".format(args.data), end="")
    data = PhysicalTensor.from_file(file=args.data).data # np.loadtxt(args.data)
    print("done")
    print("\tData shape: ",data.shape)
    
    #---------------------------------------#
    if args.replicate:
        print("\tReplicating data for each structure ... ", end="")
        data = np.tile(data,(len(atoms),1))
        print("done")
        print("\tData shape: ",data.shape)

    #---------------------------------------#
    if data.shape[0] != N and args.what in ['i','info'] :
        print("\t{:s}: shapes do not match --> keeping only the last {:d} elements of the array to be stored".format(warning,N))
        data = data[-N:]
    #---------------------------------------#
    # reshape
    Natoms = atoms[0].positions.shape[0]
    if args.what in ['a','arrays','array']:
        data = data.reshape((N,Natoms,-1))
        what = "arrays"
    elif args.what in ['i','info']:
        # data = data.reshape((N,-1))    
        what = "info"
    else:
        raise ValueError("'what' (-w,--what) can be only 'i' (info), or 'a' (array)")
    print("\tData reshaped to: ",data.shape)

    #---------------------------------------#
    # store
    print("\tStoring data to '{:s}' with name '{}' ... ".format(what,args.name), end="")
    # atoms = list(atoms)
    for n in range(N):
        if what == "info":
            atoms[n].info[args.name] = data[n]
        elif what == "arrays":
            atoms[n].arrays[args.name] = data[n]
        else:
            raise ValueError("internal error")

    #---------------------------------------#
    print("\n\tWriting trajectory to file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(file=args.output)
    print("done")

if __name__ == "__main__":
    main()

