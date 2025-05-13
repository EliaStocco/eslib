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
from mpl_toolkits.mplot3d import Axes3D

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
    os.makedirs(args.output, exist_ok=True)
    
    #------------------#
    with eslog(f"Reading input from '{args.input}'"):
        df = pd.read_csv(args.input, sep='\s+')
        
    df['R-mic'] = df['R-mic'].round(2)
    df = df.groupby(['structure', 'R-mic'], as_index=False)['autocorr'].mean()
        
    #------------------#
    with eslog("Plotting 3D scatter plot"):
        # Create the figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting the scatter points
        scatter = ax.scatter(df['R-mic'], df['structure'], df['autocorr'], c=df['structure'], cmap='viridis', marker='o')

        # Labels
        ax.set_xlabel(r'distance $\mathrm{\AA}$')
        ax.set_ylabel('time')
        # ax.set_zlabel('Autocorr')

        # Title
        ax.set_title('3D Scatter Plot: Structure vs R-mic vs Autocorr')

        # Add color bar
        fig.colorbar(scatter)

    # Show the plot
    with eslog(f"Saving plot to file '{args.output}'"):
        plt.savefig(f"{args.output}/scatter.png",dpi=300)
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
