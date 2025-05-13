#!/usr/bin/env python
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

#---------------------------------------#
description = "Analyse the autocorrelation of the unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-c" , "--cellpars"    , **argv, required=True , type=str  , help="input file with the cellpars produced by 'cellpars2txt.py'")
    # parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-a" , "--autocorr"    , **argv, required=True , type=str  , help="input file with the autocorrelation produced by 'autocorrelate-unit-cell-dipoles.py'")
    parser.add_argument("-o" , "--output"      , **argv, required=True , type=str  , help="output file")
    return parser

#---------------------------------------#
def compute_mic(args):
    s, vectors_s, cell = args
    zero = np.zeros(3)
    out, _ = get_distances(zero, vectors_s, cell, pbc=True)
    return s, out[0]

def get_mic_distance(vectors: np.ndarray, cells: List[np.ndarray]) -> np.ndarray:
    assert vectors.ndim == 3
    assert len(cells) == vectors.shape[0]

    num_structures = vectors.shape[0]
    results = np.zeros_like(vectors)
    results[:,:,:] = np.nan

    # Prepare the arguments for multiprocessing
    args = [(s, vectors[s], cells[s]) for s in range(num_structures)]

    with Pool(processes=cpu_count()) as pool:
        for s, mic in tqdm(pool.imap_unordered(compute_mic, args), total=num_structures, desc="\tComputing MIC distances",disable=True):
            results[s] = mic

    assert not np.any(np.isnan(results)), "NaN found"
    return results

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    with eslog(f"Reading input from '{args.autocorr}'"):
        autocorr = pd.read_csv(args.autocorr, sep='\s+')
        for k in ["structure","unit_cell"]:
            autocorr[k] = autocorr[k].astype(int)
        L123 = autocorr[["structure","unit_cell","L1","L2","L3"]]
        for k in ["L1","L2","L3"]:
            autocorr[k] /= max(autocorr[k])+1

    #------------------#
    with eslog(f"Reading the cellpars from file '{args.cellpars}'"):
        cellpars = np.loadtxt(args.cellpars)
        cells = [ Cell.fromcellpar(cellpar) for cellpar in cellpars ]
        npcell = np.asarray([ cell.T for cell in cells ]) # n^th column = n^th lattice vector
    
    #------------------#
    with eslog("Preparing cartesian vectors"):
        tmp_cellpars = npcell[autocorr['structure'].values]
        frac = autocorr[["L1","L2","L3"]].values
        cart = np.einsum('ijk,ik->ij',tmp_cellpars,frac)
        
        for k in ["L1","L2","L3"]:
            del autocorr[k]
            
        for n,k in enumerate(["x","y","z"]):
            autocorr[k] = cart[:,n] 
            
        data, info = pandas2ndarray(autocorr,["structure","unit_cell"],ignore_columns=["autocorr"])
        ac, _ = pandas2ndarray(autocorr,["structure","unit_cell"],ignore_columns=["x","y","z"])

    #------------------#
    with eslog("Computing MIC distances"):
        mic = get_mic_distance(data,cells)
        Rpbc = np.linalg.norm(mic,axis=2)[:,:,np.newaxis]
        mic = np.concatenate([mic,Rpbc],axis=2)
        
    #------------------#
    with eslog("Computing cartesian distances"):
        Rxyz = np.linalg.norm(data,axis=2)[:,:,np.newaxis]
        data = np.concatenate([data,Rxyz],axis=2)
        xyz_df    = melt(data,index={0:"structure",1:"unit_cell"},value_names=["x","y","z","R"])
    
    #------------------#
    with eslog("Constructing dataframes"):
        pbcxyz_df = melt(mic,index={0:"structure",1:"unit_cell"},value_names=["x-mic","y-mic","z-mic","R-mic"])
        ac_df     = melt(ac,index={0:"structure",1:"unit_cell"},value_names=["autocorr"])
        df = merge_dataframes([xyz_df,L123,pbcxyz_df,ac_df],["structure","unit_cell"])
    
    with eslog(f"Saving results to '{args.output}'"):
        df2txt(df,args.output,int_columns=["structure","unit_cell","L1","L2","L3"])
        
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
