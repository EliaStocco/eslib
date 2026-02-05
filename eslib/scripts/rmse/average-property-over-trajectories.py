#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.io_tools import pattern2sorted_files

#---------------------------------------#
# Description of the script's purpose
description = "Average properties over trajectories."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-p" , "--properties"   , **argv, required=True , type=str, help="list of properties", default=None)
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\tProcessing input pattern '{:s}' ... ".format(args.input), end="")
    files = pattern2sorted_files(args.input)
    print("done")
    print("found {:d} files.".format(len(files)))
    
    #------------------#
    trajectories = [None]*len(files)
    for n, file in enumerate(files):
        print("\tReading atomic structures from file '{:s}' ... ".format(file), end="")
        traj = AtomicStructures.from_file(file=file,input_format=args.input_format)
        print("done")
        print("\tn. of atomic structures: ",len(traj),end="\n\n")
        trajectories[n] = traj

    #------------------#
    assert trajectory.has(args.ref_name), f"Reference property '{args.ref_name}' not found in the trajectory."
    assert trajectory.has(args.pred_name), f"Predicted property '{args.pred_name}' not found in the trajectory."
    
    N = len(trajectory)
    ref = trajectory.get(args.ref_name)
    pred = trajectory.get(args.pred_name)
    natoms = np.asarray(trajectory.call(lambda x: x.get_global_number_of_atoms()))
    
    if args.what == 'energy':
        
        assert ref.ndim == 1, "Reference energy should be a 1D array."
        assert pred.ndim == 1, "Predicted energy should be a 1D"
        
        rmse = (ref - pred) / natoms # error per atom
        rmse = rmse ** 2 # squared error
        assert rmse.ndim == 1, "RMSE should be a 1D array."
        rmse = np.sqrt(np.mean(rmse)) # root mean squared error
        rmse = convert(rmse,"energy","electronvolt","millielectronvolt")
        print(f"\tRMSE of energy per atom: {rmse:.8f} [meV/atoms]")
        
    elif args.what == 'forces':
        
        for n in range(N):
            assert ref[n].ndim == 2, "Reference forces should be a 2D array."
            assert pred[n].ndim == 2, "Predicted forces should be a 2D array."
            
        ref = np.asarray(np.vstack(ref))
        pred = np.asarray(np.vstack(pred))
        
        assert ref.shape == pred.shape, "Reference and predicted forces should have the same shape."
        assert ref.shape == (np.sum(natoms).astype(int),3), "Reference forces should be a 2D array with shape (total_natoms,3)."
        
        rmse = (ref - pred) ** 2
        rmse = np.mean(rmse, axis=1) # mean over components
        assert rmse.ndim == 1, "RMSE should be a 1D array."
        rmse = np.sqrt(np.mean(rmse))
        rmse = convert(rmse,"force","ev/ang","milliev/ang")
        print(f"\tRMSE of forces: {rmse:.8f} [meV/Å]") 
    
    elif args.what == 'stress':
        
        if ref.shape == (N,3,3):
            ref = full_3x3_to_voigt_6_stress(ref)
        if pred.shape == (N,3,3):
            pred = full_3x3_to_voigt_6_stress(pred)
        
        assert ref.shape == (N,6), "Reference stress should be a 2D array with shape (N,6)."
        assert pred.shape == (N,6), "Predicted stress should be a 2D array with shape (N,6)."
        
        rmse = (ref - pred) ** 2
        rmse = np.mean(rmse, axis=1) # mean over components
        assert rmse.ndim == 1, "RMSE should be a 1D array."
        rmse = np.sqrt(np.mean(rmse))
        rmse = convert(rmse,"pressure","ev/ang3","milliev/ang3")
        print(f"\tRMSE of stress: {rmse:.8f} [meV/Å³]")
        
    elif args.what == 'dipole':
        
        assert ref.shape == (N,3), "Reference dipole should be a 2D array with shape (N,3)."
        assert pred.shape == (N,3), "Predicted dipole should be a 2D array with shape (N,3)."
        
        rmse = (ref - pred) / natoms[:,None] # error per atom
        rmse = rmse ** 2
        rmse = np.mean(rmse, axis=1) # mean over components
        assert rmse.ndim == 1, "RMSE should be a 1D array."
        _rmse = np.sqrt(np.mean(rmse))
        rmse = convert(_rmse,"electric-dipole","eang","millieang")
        print(f"\tRMSE of dipole per atom: {rmse:.8f} [meang/atoms]")
        rmse = convert(_rmse,"electric-dipole","eang","millidebye")
        print(f"\tRMSE of dipole per atom: {rmse:.8f} [mD/atoms]")
        
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()