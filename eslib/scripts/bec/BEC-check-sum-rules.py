#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.input import str2bool
from eslib.classes.models.dipole.partial_charges import DipolePartialCharges

#---------------------------------------#
# Description of the script's purpose
description = "Check that the BECs satisfy the Acoustic and Rotational Sum Rules."

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
    return parser

#---------------------------------------#
def skew_matrix(vector):
    """
    Given a 3-element array, return the corresponding skew-symmetric matrix.
    
    Parameters:
    vector (array-like): A 3-element array or list.
    
    Returns:
    np.ndarray: A 3x3 skew-symmetric matrix.
    """
    assert len(vector) == 3, "Input vector must have exactly 3 elements."
    
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def skew_matrix_general(array: np.ndarray, axis: int) -> np.ndarray:
    """
    Given an array, return the corresponding skew-symmetric matrices along the specified axis.
    
    Parameters:
    array (np.ndarray): Input array.
    axis (int): Axis along which to compute the skew-symmetric matrices. This axis must have length 3.
    
    Returns:
    np.ndarray: Array of skew-symmetric matrices.
    """
    assert array.shape[axis] == 3, "The specified axis must have length 3."
    
    # Move the specified axis to the first position
    # array = np.moveaxis(array, axis, 0)
    
    # Apply the skew_matrix function to each vector along the first axis
    return np.apply_along_axis(skew_matrix, axis, array)
    
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input,format=args.input_format)
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
    assert BEC.ndim == 3, "BEC.ndim: {:d}".format(BEC.ndim)
    assert BEC.shape == (N,3*Natoms,3), "BEC.shape: {:s}".format(BEC.shape)    
    print('\t - 1st axis: snapshot')
    print('\t - 2nd axis: dof')
    print('\t - 3rd axis: dipole xyz')

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
            # np.savetxt("TEST.txt",tmp[0],fmt=dec_format)
            
    model = DipolePartialCharges({"H":1,"O":-2},compute_BEC=True)
    test = model.compute(trajectory,raw=True)["BEC"]
    
            
    #------------------#
    # reshape
    print("\n\tReshaping the BEC tensors ... ", end="")
    BEC = BEC.reshape((N,Natoms,3,3))
    BEC = test.reshape(BEC.shape)
    print("done")
    print("\tBEC.shape: ",BEC.shape)
    assert BEC.ndim == 4, "BEC.ndim: {:d}".format(BEC.ndim)
    assert BEC.shape == (N,Natoms,3,3), "BEC.shape: {:s}".format(BEC.shape)    
    print('\t - 1st axis: snapshot')
    print('\t - 2nd axis: atom')
    print('\t - 3rd axis: atom index')
    print('\t - 4th axis: dipole xyz')
    
    #------------------#
    # Acoustic sum rule
    ASR = np.abs(np.mean(BEC,axis=1))
    print("\n\tAcoustic Sum Rule:")
    print(f"\t - max : {np.max(ASR):.4e}")
    print(f"\t - mean: {np.mean(ASR):.4e}")
    print(f"\t - std : {np.std(ASR):.4e}")
    
    #------------------#
    # Rotational sum rule
    print("\n\tExtracting the atomic positions from the trajectory ... ", end="")
    pos = trajectory.get("positions")
    print("done")
    print("\tpos.shape: ",pos.shape)
    print('\t - 1st axis: snapshot')
    print('\t - 2nd axis: atom index')
    print('\t - 3rd axis: atom xyz')
    
    print("\n\tTransforming positions to skew matrices ... ", end="")
    # pos /= np.linalg.norm(pos,axis=(1,2))[:,np.newaxis,np.newaxis]
    skew = skew_matrix_general(pos, axis=2)
    print("done")
    print("\tskew.shape: ",skew.shape)
    print('\t - 1st axis: snapshot')
    print('\t - 2nd axis: atom index')
    print('\t - 3rd axis: skew matrix xyz')
    print('\t - 4th axis: skew matrix xyz')
    
    RSM = np.einsum("ijkl,ijlm->ijkm", skew, BEC)
    assert np.allclose(skew @ BEC, RSM), "coding error"
    RSM = np.abs(np.mean(RSM,axis=1))
    # norm = 1./np.linalg.norm(pos,axis=1)
    # RSM = RSM * norm[:,:,np.newaxis]
    print("\n\tRotational Sum Rule:")
    print(f"\t - max : {np.max(RSM):.4e}")
    print(f"\t - mean: {np.mean(RSM):.4e}")
    print(f"\t - std : {np.std(RSM):.4e}")
    
    pass
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()