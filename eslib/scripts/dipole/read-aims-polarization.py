#!/usr/bin/env python
import re
import numpy as np
import glob
import pandas as pd
from eslib.tools import convert
from eslib.formatting import esfmt, float_format, warning
from typing import Tuple
from ase.io import read
from eslib.functions import check_pattern_in_file
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Extract the values of the polarization from a bunch of FHI-aims files and compute the dipoles."

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
def read_polarization(file: str) -> Tuple[np.ndarray, str]:
    """
    Read the polarization values from a file.

    Args:
        file (str): The path to the file to be read.

    Returns:
        Tuple[np.ndarray, str]: A tuple containing the polarization values as a NumPy array and the line containing the polarization.
    """
    # Initialize the polarization array with NaN values
    polarization = np.full(3, np.nan)
    line_with_polarization = ""

    with open(file, 'r') as f:
        for line in f:
            if "| Cartesian Polarization" in line:
                match_obj = re.search(r'\| Cartesian Polarization\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)', line)
                if match_obj:
                    polarization = np.array([float(match_obj.group(1)), float(match_obj.group(2)), float(match_obj.group(3))])
                    line_with_polarization = line.strip()
                    break

    return polarization, line_with_polarization

#---------------------------------------#
def extract_number_from_filename(filename: str) -> float:
    """
    Extract the first numerical value found in a filename.

    Args:
        filename (str): The filename from which to extract the numerical value.

    Returns:
        float: The numerical value extracted from the filename.
    """
    # Use regular expressions to find the first numerical value in the filename
    match_obj = re.search(r'(\d+\.?\d*)', filename)
    if match_obj:
        return float(match_obj.group(1))
    return float('inf')  # If no number is found, return infinity so these files appear last

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print("\n\tSearching for files ... ", end="")
    all_files = glob.glob(args.input)
    all_files = sorted(all_files, key=lambda x: (x, extract_number_from_filename(x)))
    print("done")
    N = len(all_files)
    print("\tn. files: ", N)

    #------------------#
    print("\n\tKeeping only files with 'Have a nice day' ... ", end="")
    finished = [ check_pattern_in_file(f,'Have a nice day') for f in all_files ]
    all_files = [ f for f, finished_flag in zip(all_files, finished) if finished_flag]
    print("done")
    N = len(all_files)
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
    columns = ["file", "Px [au]", "Py [au]", "Pz [au]", "P [au]", "volume [au]", "dx [au]", "dy [au]", "dz [au]", "d [au]","string"]
    df = pd.DataFrame(columns=columns, index=np.arange(N))

    structures = [None]*N
    good = np.full(N,fill_value=True,dtype=bool)
    
    #------------------#
    print("\n\tReading files and extracting the polarization ... ", end="")
    for n, file in enumerate(all_files):
        df.at[n, "file"] = file

        P, string = read_polarization(file) # C/m^2
        try :
            structure = read(file,format='aims-output')
        except:
            print("\tskipping file '{:s}'".format(file))
            good[n] = False
            continue

        structures[n] = structure

        V = structure.get_volume() # angstrom^3
        V = convert(V,"volume","angstrom3","atomic_unit")
        P = convert(P,"polarization","C/m^2","atomic_unit")
        dipole = P * V

        df.at[n, "Px [au]"] = P[0]
        df.at[n, "Py [au]"] = P[1]
        df.at[n, "Pz [au]"] = P[2]
        df.at[n, "P [au]"] = np.linalg.norm(P)
        df.at[n, "volume [au]"] = V

        df.at[n, "dx [au]"] = dipole[0]
        df.at[n, "dy [au]"] = dipole[1]
        df.at[n, "dz [au]"] = dipole[2]
        df.at[n, "d [au]"] = np.linalg.norm(dipole)

        df.at[n,"string"] = string

    print("done")

    #------------------#
    dipoles = df[ ["dx [au]","dy [au]","dz [au]"]]
    dipoles = np.asarray(dipoles).astype(float)
    assert dipoles.shape == (N,3)
    dipoles = convert(dipoles,"electric-dipole","atomic_unit",args.unit)
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
