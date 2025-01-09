#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import convert

#---------------------------------------#
description = "Evaluate a the RMSE per atom of the dipole between two datasets." 

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i", "--input"     , **argv, type=str, required=False , help='input extxyz file (default: %(default)s)',default=None )
    parser.add_argument("-e", "--expected"  , **argv, type=str, required=False , help="keyword or txt file with the expected values (default: %(default)s)", default="exp.txt")
    parser.add_argument("-p", "--predicted" , **argv, type=str, required=False , help="keyword or txt file with the predicted values (default: %(default)s)", default="pred.txt")
    return parser

@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    if args.input is not None:
        print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
        atoms = AtomicStructures.from_file(file=args.input,format="extxyz")
        print("done")
        Natoms = atoms[0].get_global_number_of_atoms()

        predicted = atoms.get(args.predicted)
        expected = atoms.get(args.expected)

    else:
        #------------------#
        print("\tReading predicted values from file '{:s}' ... ".format(args.predicted), end="")
        predicted = np.loadtxt(args.predicted)
        print("done")
        

        #------------------#
        print("\tReading expected values from file '{:s}' ... ".format(args.expected), end="")
        expected = np.loadtxt(args.expected)
        print("done")
        

    print("\tpredicted.shape: ",predicted.shape)
    print("\texpected.shape: ",expected.shape)

    assert predicted.ndim == 2, "the predicted dipoles must be a 2D array"
    assert expected.ndim == 2, "the expected dipoles must be a 2D array"
    assert predicted.ndim == expected.ndim 
    assert predicted.shape == expected.shape 
    assert predicted.shape[1] == 3, "the dipoles must have 3 components"
    
    #------------------#
    print("\tComputing metrics ... ", end="")
    diff  = predicted - expected        
    err_atoms = diff/Natoms
    err_atoms_2 = np.square(err_atoms)
    diff2 = err_atoms_2.sum(axis=1)      # sum over x,y,z
    mean = np.mean(diff2)                # mean over snapshots
    rmse = np.sqrt(mean)
    print("done")
    
    rmse *= 1000
    print("\tRMSE (meAng): ",rmse)
    
    rmse = convert(rmse,"electric-dipole","millieang","millidebye")
    print("\tRMSE (mDebye): ",rmse)
    
#---------------------------------------#
if __name__ == "__main__":
    main()
