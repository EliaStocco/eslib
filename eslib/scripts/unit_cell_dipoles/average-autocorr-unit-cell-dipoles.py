#!/usr/bin/env python
import numpy as np
import pandas as pd
from ase.cell import Cell
from eslib.formatting import esfmt, eslog
from eslib.mathematics import mean_std_err
from eslib.dataframes import df2txt
from eslib.io_tools import pattern2sorted_files

#---------------------------------------#
description = "Compute mean and standard deviation of the unit-cells dipoles autocorrelation."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--input" , **argv, required=True , type=str  , help="input files produced by 'analyse-autocorrelate-unit-cell-dipoles.py'")
    parser.add_argument("-o" , "--output", **argv, required=True , type=str  , help="output file prefix")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    files = pattern2sorted_files(args.input)
    print(f"\tFound {len(files)} files to process.\n")

    #------------------#
    dfs = [None] * len(files)
    for i, file in enumerate(files):
        try:
            print(f"\tReading file {i+1}/{len(files)}: {file} ...", end="")
            dfs[i] = pd.read_csv(file, sep="\s+")
            print("done")
        except Exception as e:
            print("")
            raise ValueError(f"Error reading file {file}: {e}")
    
    #------------------#
    # Columns to check for consistency
    check_columns = ["structure", "unit_cell", "x", "y", "z", "x-mic", "y-mic", "z-mic", "R-mic"]
    
    reference_df = dfs[0][check_columns].copy()

    for i, df in enumerate(dfs[1:], start=1):
        current_df = df[check_columns].copy()
        if not reference_df.equals(current_df):
            raise ValueError(f"Data mismatch found in file {files[i]} compared to {files[0]}")
    
    print("\n\t✅ All dataframes have consistent data in the specified columns.")

    #------------------#
    # Extract all autocorr columns into a 2D NumPy array
    print("\n\tExtracting autocorrelation data from all dataframes ...", end="")
    autocorr_matrix = np.array([df["autocorr"].values for df in dfs])
    print("done")
    print(f"\tautocorrelation.shape: {autocorr_matrix.shape}")

    # Compute mean, std, stderr along axis=0 (across files)
    print("\n\tComputing mean, standard deviation, and standard error of the mean for autocorrelation ...", end="")
    autocorr_mean, autocorr_std, autocorr_stderr = mean_std_err(autocorr_matrix, axis=0)
    print("done")

    #------------------#
    # Create DataFrames with metadata + stats
    metadata = dfs[0].drop(columns=["autocorr"]).copy()

    df_mean = metadata.copy()
    df_mean["autocorr"] = autocorr_mean

    df_std = metadata.copy()
    df_std["autocorr"] = autocorr_std

    df_stderr = metadata.copy()
    df_stderr["autocorr"] = autocorr_stderr

    print("\t✅ Created mean, std, and stderr DataFrames.")
    
    #------------------#
    ofile = args.output + "-mean.txt"
    print(f"\n\tSaving mean to file  '{ofile}' ... ",end="")
    df2txt(df_mean,ofile,int_columns=["structure","unit_cell","L1","L2","L3"])
    print("done")
    
    ofile = args.output + "-std.txt"
    print(f"\n\tSaving std to file  '{ofile}' ... ",end="")
    df2txt(df_std,ofile,int_columns=["structure","unit_cell","L1","L2","L3"])
    print("done")
    
    ofile = args.output + "-stderr.txt"
    print(f"\n\tSaving stderr to file  '{ofile}' ... ",end="")
    df2txt(df_stderr,ofile,int_columns=["structure","unit_cell","L1","L2","L3"])
    print("done")
    
    return    

#---------------------------------------#
if __name__ == "__main__":
    main()
