#!/usr/bin/env python
import argparse
import pickle

import numpy as np
import xarray as xr

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt, warning
from eslib.input import str2bool
from eslib.output import output_folder
from eslib.tools import convert
from eslib.units import remove_unit

# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
# Description of the script's purpose
description = "Project a trajectory onto the normal modes."
documentation = "The positions and lattice vectors are supposed to be in angstrom while the velocities are supposed to be in atomic units."

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    # parser.add_argument("-g",  "--ground_state", type=str, **argv, help="ground-state atomic structure [a.u.] (default: %(default)s)", default="start.xyz")
    parser.add_argument("-t" ,  "--trajectory"   , type=str     , **argv, required=True , help="input extxyz file [ang]")
    parser.add_argument("-nm", "--normal_modes"  , type=str     , **argv, required=True , help="normal modes file")
    parser.add_argument("-n"  , "--indices"      ,   **argv, required=False, type=str, help="*.txt file with the indices")
    parser.add_argument("-r", "--use_reference"  , type=str2bool, **argv, required=False, help="use NormalModes reference structure instead of the first structure of the trajectory' (default: %(default)s)", default=True)
    parser.add_argument("-o" ,  "--output"       , type=str     , **argv, required=False, help="output file  (default: %(default)s)", default=None)
    parser.add_argument("-of",  "--output_folder", type=str     , **argv, required=False, help="output folder for csv files (default: %(default)s)", default="project")
    return parser

@esfmt(prepare_args,description,documentation)
def main(args):

    #---------------------------------------#
    print("\tReading phonon modes from file '{:s}' ... ".format(args.normal_modes), end="")
    with open(args.normal_modes,'rb') as f:
        nm = pickle.load(f)
    print("done")

    if type(nm) != NormalModes:
        raise TypeError("Loaded object is of wrong type, it should be a 'NormalModes' object")
    
    if nm.reference is None:
        print(f"\t{warning}: the normal modes file does not have a reference structure.\n\tThe first structure of the trajectory will be used as reference.")
    else:
        print("\tThe normal modes file has a reference structure.")
        
    if not args.use_reference:
        nm.reference = None
    
    #---------------------------------------#
    print("\n\tReading trajectory from file '{:s}' ... ".format(args.trajectory), end="")
    trajectory = AtomicStructures.from_file(file=args.trajectory)
    print("done")
    
    factor_pos = convert(what=1,family="length",_from="angstrom",_to="atomic_unit")
    print(f"\tConverting positions and lattice vectors from 'angstrom' to 'atomic_unit' using factor {factor_pos} ... ",end="")
    for n,atoms in enumerate(trajectory):
        # print("\t{:d}/{:d} ... ".format(n+1,N), end="\r")
        atoms.positions *= factor_pos
        if np.all(atoms.get_pbc()):
            atoms.cell *= factor_pos
            
    #------------------#
    if args.indices is not None:
        print("\tReading indices from file '{:s}' ... ".format(args.indices), end="")
        indices = np.loadtxt(args.indices,dtype=int).astype(int)
        print("done")
        
        #------------------#
        print("\n\tSorting the following arrays: ", trajectory.get_keys("arrays"))
        print("\tSorting ... ",end="")
        N = len(trajectory)
        for n in range(len(trajectory)):
            trajectory[n] = trajectory[n][indices]
        print("done")

    #---------------------------------------#
    # project on phonon modes
    print("\n\tProjecting the trajectory ... ",end="")
    results = nm.project(trajectory)
    print("done")
    
    #---------------------------------------#
    # save result to file
    if args.output is not None:
        print("\n\tWriting results to file '{:s}' in pickle format ... ".format(args.output), end="")
        # Open the file in binary write mode ('wb')
        with open(args.output, 'wb') as file:
            # Use pickle.dump() to serialize and save the object to the file
            pickle.dump(results, file)
        print("done")
    
    #---------------------------------------#
    if args.output_folder is not None:
        
        comment_lines = [
            "# column: normal mode",
            "# row: MD step"
        ]
        
        print("\n\tWriting results to folder '{:s}' in separated csv files:".format(args.output_folder))
        output_folder(args.output_folder,show=False)
        for k in results.keys():
            arr = xr.DataArray(results[k])
            file = "{:s}/{:s}.csv".format(args.output_folder,k)
            print("\t\tsaving '{:s}' to file '{:s}' ... ".format(k,file), end="")
            arr = remove_unit(arr)[0]
            df = arr.T.to_pandas()
            
            # Open the file in write mode
            with open(file, 'w') as f:
                # Write the comments first
                for line in comment_lines:
                    f.write(line + "\n")
                df.to_csv(f,index=False,header=False,na_rep="nan",float_format="%24.16f")
            print("done")

        print("\n\tHow to read the csv files:")
        print("\t\tcolumns: normal mode index")
        print("\t\t   rows: time step")

if __name__ == "__main__":
    main()