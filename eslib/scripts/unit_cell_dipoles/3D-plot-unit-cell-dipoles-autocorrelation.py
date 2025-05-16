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
    df:pd.DataFrame = df.groupby(['structure', 'R-mic'], as_index=False)['autocorr'].mean()
        
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
        plt.close()
        
    #------------------#
    # Sort and prepare the pivoted data
    # Prepare pivot table
    heatmap_data = df.pivot(index='structure', columns='R-mic', values='autocorr')
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)

    x = np.array(heatmap_data.columns)
    y = np.array(heatmap_data.index)
    z = heatmap_data.values

    # Reduce tick density if too many labels
    max_ticks = 40
    x_tick_step = max(1, len(x) // max_ticks)
    y_tick_step = max(1, len(y) // max_ticks)

    # Use numerical indices for y to avoid slow string tick drawing
    y_numeric = np.arange(len(y))

    with eslog("Plotting optimized 2D heatmap"):
        fig, ax = plt.subplots(figsize=(12, 8))

        X, Y = np.meshgrid(x, y_numeric)

        c = ax.pcolormesh(X, Y, z, shading='nearest', cmap='viridis')  # Faster than 'auto'

        # Optimize ticks
        ax.set_xticks(x[::x_tick_step])
        ax.set_xticklabels([f"{val:.2f}" for val in x[::x_tick_step]], rotation=90)

        ax.set_yticks(y_numeric[::y_tick_step])
        ax.set_yticklabels(y[::y_tick_step])

        ax.set_xlabel('R-mic (Å)')
        ax.set_ylabel('Structure')
        ax.set_title('Heatmap of Autocorrelation by Structure and Distance')

        fig.colorbar(c, ax=ax, label='Autocorr')

        with eslog(f"Saving heatmap to file '{args.output}/heatmap.png'"):
            plt.tight_layout()
            plt.savefig(f"{args.output}/heatmap.png", dpi=300)
            plt.close()



    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
