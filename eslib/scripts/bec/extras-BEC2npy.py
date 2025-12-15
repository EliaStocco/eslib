#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
from eslib.input import itype
from eslib.classes.aseio import integer_to_slice_string

#---------------------------------------#
# Description of the script's purpose
description = "Convert the BECs from an extras i-PI file to a npy file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="atomic structures file [extxyz]")
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output file with the BEC tensors (default: %(default)s)", default='bec.txt')
    return parser# .parse_args()
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading Born Effective Charges from file '{:s}' ... ".format(args.input), end="")
    Z = np.loadtxt(args.input)
    print("done")
    # print("\tZ.shape: ",Z.shape)
    
    #------------------#
    index = integer_to_slice_string(args.index)
    Z = Z[index,:]
    print("\tZ.shape: ",Z.shape)
    
    #------------------#
    Nstructures = Z.shape[0]
    Natoms = int(Z.shape[1] / 9)
    print("\n\tnumber of structures: ",Nstructures)
    print("\tnumber of atoms:      ",Natoms)
    Z = Z.reshape((Nstructures,Natoms,3,3))
    print("\tZ.shape: ",Z.shape)
    
    #------------------#
    # acoustic sum rule
    acr = np.sum(Z,axis=1,keepdims=True) / Natoms
    check = np.allclose(acr,0)
    print("\n\tacoustic sum rule check: ",check)

    #------------------#
    print("\tSaving Born Effective Charges to file '{:s}' ... ".format(args.output), end="")
    np.save(args.output,Z)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()