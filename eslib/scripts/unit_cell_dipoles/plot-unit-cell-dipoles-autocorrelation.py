#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from typing import List
from ase.cell import Cell
from eslib.formatting import esfmt, eslog
from eslib.mathematics import pandas2ndarray, melt, merge_dataframes
from eslib.dataframes import df2txt
from ase.geometry import get_distances
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import matplotlib.pyplot as plt

#---------------------------------------#
description = "Plot the autocorrelation of the unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--input"    , **argv, required=True , type=str  , help="input file produced by 'analyse-autocorrelate-unit-cell-dipoles.py'")
    parser.add_argument("-o" , "--output"   , **argv, required=True , type=str  , help="output folder")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    os.makedirs(args.output,exist_ok=True)
    
    #------------------#
    with eslog(f"Reading input from '{args.input}'"):
        df = pd.read_csv(args.input, sep='\s+')
        
    #------------------#
    # fig, ax = plt.subplots(figsize=(4,4))
    # structures = np.unique(df["structure"])
    # cmap = plt.cm.get_cmap("viridis", len(structures)) 
    # for n,s in tqdm(enumerate(structures), total=len(structures), desc="\tProcessing structures"):
    #     sub_df = df.loc[df["structure"] == s,["R-mic","autocorr"]]
    #     ax.scatter(sub_df["R-mic"], sub_df["autocorr"], label=s, color=cmap(n))
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(4,4))
    
    structures = np.unique(df["structure"])
    
    cmap = plt.cm.get_cmap("viridis", len(structures)) 

    # Normalize the 'structure' values so that they fit the colormap range
    norm = plt.Normalize(vmin=0, vmax=40000)  # You can adjust the `vmax` depending on the actual max value of structures.

    # Use tqdm to monitor progress in the loop
    for structure, sub_df in tqdm(df.groupby("structure"), total=len(df["structure"].unique()), desc="\tProcessing structures"):
        color = cmap(norm(structure))  # Map the structure index to a color
        ax.scatter(sub_df["R-mic"], sub_df["autocorr"], color=color)

    # Show the plot
    plt.show()
        
    
        
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
