#!/usr/bin/env python
import glob
import re
from typing import Tuple

import numpy as np
import pandas as pd
from ase.io import read

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format, warning
from eslib.functions import check_pattern_in_file, extract_number_from_filename, extract_float
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Extract the values of the dipole from a bunch of FHI-aims files."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--input"            , **argv, type=str, required=True , help="input file, folder, or search options")
    parser.add_argument("-u" , "--unit"             , **argv, type=str, required=False, help="output dipoles unit (default: %(default)s)", default='eang')
    parser.add_argument("-bf", "--bad_files"        , **argv, type=str, required=False, help="txt output file with the filepath of the non-converged calculations (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"           , **argv, type=str, required=False, help="output file with the dipole values (default: %(default)s)", default="dipole.eang.txt")
    parser.add_argument("-oi", "--output_info"      , **argv, type=str, required=False, help="*.csv output file with information (default: %(default)s)", default="info.csv")
    parser.add_argument("-os", "--output_structures", **argv, type=str, required=False, help="output file with the dipole values and the atomic structures (default: %(default)s)", default='aims-with-dipoles.ang.extxyz')
    parser.add_argument("-of", "--output_format"    , **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    parser.add_argument("-k", "--keyword"           , **argv, type=str, required=False, help="keyword of the dipoles (default: %(default)s)", default='dipole')
    return parser

#---------------------------------------#
def read_dipole(file: str) -> Tuple[np.ndarray, str]:
    """
    Read the dipole values from a file.

    Args:
        file (str): The path to the file to be read.

    Returns:
        Tuple[np.ndarray, str]: A tuple containing the dipole values as a NumPy array and the line containing the dipole.
    """
    # Initialize the dipole array with NaN values
    dipole = np.full(3, np.nan)
    line_with_dipole = ""

    with open(file, 'r') as f:
        for line in f:
            if "| Total dipole moment" in line:
                dipole = extract_float(line)
                assert len(dipole) == 3, "wrong number of float extracted from the string"
                line_with_dipole = line.strip()
                break

    return dipole, line_with_dipole


#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print("\n\tSearching for files ... ", end="")
    all_files = glob.glob(args.input)
    all_files = [ all_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in all_files ])) ]
    print("done")
    N = len(all_files)
    print("\tn. files: ", N)

    #------------------#
    print("\n\tKeeping only files with 'Have a nice day' ... ", end="")
    finished = [ check_pattern_in_file(f,'Have a nice day') for f in all_files ]
    all_good_files = [ f for f, finished_flag in zip(all_files, finished) if finished_flag]
    print("done")
    N = len(all_good_files)
    print("\tn. files: ", N)

    #------------------#
    if args.bad_files is not None:
        print("\n\tSaving non-converged calculations to file '{:s}' ... ".format(args.bad_files), end="")
        bad_files = [ f for f, finished_flag in zip(all_files, finished) if not finished_flag]
        # Open the file in write mode and write each string to the file
        with open(args.bad_files, "w") as f:
            for string in bad_files:
                f.write(string + "\n")
    print("done")


    #------------------#
    columns = ["file", "dx [eang]", "dy [eang]", "dz [eang]", "d [eang]","string"]
    df = pd.DataFrame(columns=columns, index=np.arange(N))

    structures = [None]*N
    good = np.full(N,fill_value=True,dtype=bool)
    
    #------------------#
    print("\n\tReading files and extracting the dipole ... ", end="")
    for n, file in enumerate(all_good_files):
        df.at[n, "file"] = file

        
        try :
            structure = read(file,format='aims-output')
            assert np.all(~structure.get_pbc()), "The atomic structures should not be periodic"
        except:
            print("\tskipping file '{:s}'".format(file))
            good[n] = False
            continue

        dipole, string = read_dipole(file) # C/m^2

        assert np.any(~np.isnan(dipole)), "Found NaN values"

        structures[n] = structure

        df.at[n, "dx [eang]"] = dipole[0]
        df.at[n, "dy [eang]"] = dipole[1]
        df.at[n, "dz [eang]"] = dipole[2]
        df.at[n, "d [eang]"] = np.linalg.norm(dipole)

        df.at[n,"string"] = string

    print("done")

    #------------------#
    dipoles = df[ ["dx [eang]","dy [eang]","dz [eang]"]]
    dipoles = np.asarray(dipoles).astype(float)
    assert dipoles.shape == (N,3)
    dipoles = convert(dipoles,"electric-dipole","eang",args.unit)
    print("\n\tSaving dipoles to file '{:s}' ... ".format(args.output), end="")
    file = str(args.output)
    if file.endswith("txt"):
        np.savetxt(file,dipoles,fmt=float_format)
    elif file.endswith("npy"):
        np.save(file,dipoles)
    print("done")

    #------------------#
    print("\n\tSaving information to file '{:s}' ... ".format(args.output_info), end="")
    df.to_csv(args.output_info, index=False)
    print("done")

    #------------------#
    if np.any(~good) :
        print("\n\t{:s}: there are {:s} bad structures --> they will be discarded".format(warning,np.sum(~good)))
        file = 'bad-structures-indices.txt'
        print("\tSaving indices of the bad structures to file 'bad-structures-indices.txt' ... ".format(file), end="")
        indices = np.where(~good)
        np.savetxt(file,indices,fmt='%d')
        print("done")
    
    #------------------#
    print("\n\tPreparing atomic structures ... ", end="")
    structures = [ s for s,g in zip(structures,good) if g ]
    structures = AtomicStructures(structures)
    structures.set(args.keyword,dipoles,"info")
    print("done")

    #------------------#
    print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.output_structures), end="")
    structures.to_file(file=args.output_structures,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
