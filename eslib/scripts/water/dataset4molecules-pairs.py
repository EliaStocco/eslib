#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from ase.cell import Cell
from classes.atomic_structures import PARALLEL
from eslib.formatting import esfmt, eslog
from ase.geometry import find_mic
from multiprocessing import Pool

PARALLEL = False  # Enable parallel processing

#---------------------------------------#
# Description of the script's purpose
description = "Create a dataset for each molecules pair."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="csv produced by 'aggregate-into-molecules.py'")
    parser.add_argument("-c" , "--cells"        , **argv, required=True , type=str, help="file with cells")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output folder")
    return parser

#---------------------------------------#
def process_pair(i, j, df_i, df_j, cellpars, npcell, output_dir):

    assert len(df_i) == len(cellpars), "Mismatch in structures vs cellpars for i"
    assert len(df_j) == len(cellpars), "Mismatch in structures vs cellpars for j"

    df_ij = pd.merge(df_i, df_j, on="time", suffixes=('_i', '_j'))

    delta = np.asarray([
        df_ij["Rx_i"] - df_ij["Rx_j"],
        df_ij["Ry_i"] - df_ij["Ry_j"],
        df_ij["Rz_i"] - df_ij["Rz_j"]
    ]).T  # Shape (N, 3)

    distances = np.empty(len(cellpars))
    for n in range(len(cellpars)):
        _, dist = find_mic(delta[n], cell=npcell[n], pbc=True)
        distances[n] = dist

    df_ij["distance"] = distances

    for col in ["molecule_i", "molecule_j", "Rx_i", "Ry_i", "Rz_i", "Rx_j", "Ry_j", "Rz_j"]:
        if col in df_ij:
            del df_ij[col]

    ofile = f"{output_dir}/pair-{i}-{j}.csv"
    df_ij.to_csv(ofile, index=False)
    # return f"pair {i}-{j} done"

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # data
    with eslog(f"Reading data fram file '{args.input}'"):
        df = pd.read_csv(args.input)
    
    #------------------#
    # cells
    with eslog(f"Reading the cellpars from file '{args.cells}'"):
        cellpars = np.loadtxt(args.cells)
        cells = [ Cell.fromcellpar(cellpar) for cellpar in cellpars ]
        npcell = np.asarray([ cell.T for cell in cells ]) # n^th column = n^th lattice vector
    
    #------------------#
    # inter-molecular distances
    molecules = np.unique(df["molecule"])
    np.sort(molecules)  # sort molecules for consistent output
    assert np.allclose(molecules, np.arange(len(molecules))), \
        "Molecules indices must be consecutive integers starting from 0."

    with eslog(f"Computing the inter-molecular distances"):
        os.makedirs(args.output, exist_ok=True)
        # Prepare args for multiprocessing
        

        if PARALLEL:
            tasks = [(i, j, df[df["molecule"] == i].copy(), df[df["molecule"] == j].copy(), cellpars, npcell, args.output) 
                     for i in range(len(molecules)) 
                     for j in range(i, len(molecules))]
            with Pool(processes=min(4,os.cpu_count())) as pool:
                pool.map(process_pair, tasks)
        else:
            for i in range(len(molecules)):
                for j in range(i, len(molecules)):
                    df_i = df[df["molecule"] == i].copy()
                    df_j = df[df["molecule"] == j].copy()
                    process_pair(i, j, df_i, df_j, cellpars, npcell, args.output)
            
    return
            
#---------------------------------------#
if __name__ == "__main__":
    main()


