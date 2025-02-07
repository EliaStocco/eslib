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
    parser.add_argument("-i", "--input"    , **argv, type=str, required=True , help='input extxyz file')
    parser.add_argument("-e", "--expected" , **argv, type=str, required=True , help="keyword or txt file with the expected values")
    parser.add_argument("-p", "--predicted", **argv, type=str, required=True , help="keyword or txt file with the predicted values")
    parser.add_argument("-c", "--charges"  , **argv, type=str, required=False, help="JSON file with the charges (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
def compute_rmse(expected:np.ndarray,predicted:np.ndarray,Natoms:np.ndarray):
    """
    Compute the RMSE with per-atom normalization.

    Parameters:
    expected : (N_snapshots, 3) np.ndarray - Reference atomic coordinates.
    predicted : (N_snapshots, 3) np.ndarray - Predicted atomic coordinates.
    Natoms : (N_snapshots,) np.ndarray - Number of atoms per snapshot.

    Returns:
    float - Root Mean Square Error (RMSE).
    """
    # \text{RMSE} = \sqrt{\frac{1}{N_s} \sum_{i=1}^{N_s} \left\| \frac{\mathbf{p}_i - \mathbf{e}_i}{N_i} \right\|^2 }
    assert expected.shape == predicted.shape, "error"
    assert expected.ndim == 2, "error"
    assert Natoms.ndim == 1, "error"
    diff  = predicted - expected        
    err_atoms = diff/Natoms[:,None]
    diff2 = np.linalg.norm(err_atoms,axis=1)**2 # norm squared over x, y, z
    mean = np.mean(diff2)                       # mean over snapshots
    rmse = np.sqrt(mean)
    return rmse

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    # if args.input is not None:
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format="extxyz")
    print("done")
    Natoms =  np.asarray([ a.get_global_number_of_atoms() for a in atoms ])

    predicted = atoms.get(args.predicted)
    expected = atoms.get(args.expected)

    # else:
    #     #------------------#
    #     print("\tReading predicted values from file '{:s}' ... ".format(args.predicted), end="")
    #     predicted = np.loadtxt(args.predicted)
    #     print("done")
        

    #     #------------------#
    #     print("\tReading expected values from file '{:s}' ... ".format(args.expected), end="")
    #     expected = np.loadtxt(args.expected)
    #     print("done")
        

    print("\tpredicted.shape: ",predicted.shape)
    print("\texpected.shape: ",expected.shape)

    assert predicted.ndim == 2, "the predicted dipoles must be a 2D array"
    assert expected.ndim == 2, "the expected dipoles must be a 2D array"
    assert predicted.ndim == expected.ndim 
    assert predicted.shape == expected.shape 
    assert predicted.shape[1] == 3, "the dipoles must have 3 components"
    
    #------------------#
    print("\tComputing metrics ... ", end="")
    rmse = compute_rmse(expected,predicted,Natoms)
    print("done")
    rmse *= 1000
    print("\tRMSE (meAng/atoms): ",rmse)
    rmse = convert(rmse,"electric-dipole","millieang","millidebye")
    print("\tRMSE (mDebye/atoms): ",rmse)
    
    #------------------#
    # adding oxidation numbers contribution
    if args.charges is not None:
        
        from eslib.classes.models.dipole import DipolePartialCharges
        import json
        with open(args.charges, 'r') as json_file:
            charges:dict = json.load(json_file)
        model = DipolePartialCharges(charges)
        baseline = model.compute(atoms,raw=True)["dipole"]
        
        #------------------#
        print("\tComputing metrics with the oxidation numbers contribution ... ", end="")
        rmse = compute_rmse(expected+baseline,predicted+baseline,Natoms)
        print("done")
        rmse *= 1000
        print("\tRMSE with oxn (meAng/atoms): ",rmse)
        rmse = convert(rmse,"electric-dipole","millieang","millidebye")
        print("\tRMSE with oxn (mDebye/atoms): ",rmse)
        
#---------------------------------------#
if __name__ == "__main__":
    main()
