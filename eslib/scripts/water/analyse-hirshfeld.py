#!/usr/bin/env python
import numpy as np
from typing import List
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format, warning
from eslib.plot import legend_options
from eslib.geometry import max_mic_distance
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------#
# Description of the script's purpose
description = "Analyse the results from 'displace-first-water-molecule.py' and 'aims-read-hirshfeld.py'."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"      , **argv, required=True , type=str  , help="forces keyword")
    parser.add_argument("-t" , "--tag"          , **argv, required=True , type=int  , help="tag")
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str  , help="molecule keyword (default: %(default)s", default="molecule")
    parser.add_argument("-p" , "--plot"         , **argv, required=False, type=str  , help="output plot file (default: %(default)s)", default="hessian.pdf")
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str  , help="output file (default: %(default)s)", default="hessian.csv")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print(f"\tReading atomic structures from file '{args.input}' ... ",end="")
    trajectory:List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    Ns = len(trajectory)
    print("\t Number atomic structures: ",Ns)
    if Ns != 19:
        print(f"\t{warning}: file incomplete, the number of structures is not 19.")
        Ns = Ns if Ns % 2 == 1 else Ns - 1
        print(f"\t{warning}: considering only the first {Ns} structures.")
        trajectory = AtomicStructures(trajectory[:Ns])
    
    #------------------#
    print("\n\tExtracting positions ... ",end="")
    positions = trajectory.get("positions")
    print("done")
    print("\tpositions.shape:", positions.shape)
    
    #------------------#
    print("\n\tChecking which atoms have been displaced ... ",end="")
    tmp = positions.reshape((positions.shape[0],-1))
    tmp = np.diff(tmp,axis=0)
    
    # Find columns with at least one non-zero element
    nonzero_cols = np.where(tmp.any(axis=0))[0]
    if len(nonzero_cols) != 9:
        print(f"\t{warning}: the number of non-zero columns is not 9")
    
    nonzero_atoms = np.unique(nonzero_cols//3).astype(np.int32)
    if len(nonzero_atoms) != 3:
        print(f"\t{warning}: the number of non-zero atoms is not 3")
    
    tmp = np.diff(positions[:,nonzero_atoms,:],axis=0).reshape((Ns-1,-1))
    if not np.all(tmp.any(axis=0)):
        print(f"\t{warning}: there is at least one column with all zero elements")
    
    print("done")
    print("\tdisplaced atoms:", nonzero_atoms)
    
    original:Atoms = trajectory[0]
    symbols = original.get_chemical_symbols()
    symbols = [symbols[i] for i in nonzero_atoms]
    print("\tdisplaced species:", symbols)
    
    #------------------#
    print("\n\tChecking displacement step ... ",end="")
    delta_R = np.abs(np.unique(positions[2]-positions[1])).max()/2
    delta_R = np.round(delta_R, decimals=6)
    print("done")
    print("\tdisplacement step [Ang]:", delta_R)
    
    #------------------#
    print("\n\tComputing interatomic distances (first snapshot only) ... ",end="")
    distances = {}
    for atom in nonzero_atoms:
        dist = original.get_distances(atom, range(0,original.get_global_number_of_atoms()), mic=True)
        distances[atom] = dist
    print("done")
    
    #------------------#
    print("\n\tExtracting molecule index ... ",end="")
    molecule = original.arrays[args.molecule].astype(np.int32)
    print("done")
    print("\tmolecule.shape:", molecule.shape)
    
    #------------------#
    print("\n\tExtracting forces ... ",end="")
    forces = trajectory.get(args.keyword)
    print("done")
    print("\tforces.shape:", forces.shape)
    if forces.ndim == 2:
        forces = forces[:,:,None]
    forces = forces[1:,:,:]
    
    plus  = forces[0::2,:,:]
    minus = forces[1::2,:,:]
    
    delta_F = (plus - minus)/2./delta_R
    hessian = np.linalg.norm(delta_F,axis=2)
    
    k = 0
    colors = ["red" if a == "O" else "blue" for a in symbols]
    
    #------------------#
    # Prepare lists to store data
    df_list = []

    # Create empty DataFrame
    df = pd.DataFrame(columns=["distance", "hessian", "displaced-atom","atom","molecule","delta_F_x","delta_F_y","delta_F_z"])

    for n, atom in enumerate(nonzero_atoms):
        x = distances[atom]
        for k in range(3*n, 3*n+3):
            if k >= len(hessian):
                break
            y = hessian[k, :]
            ii = x != 0
            # Create temporary DataFrame for this slice
            temp_df = pd.DataFrame({
                "distance": x[ii],
                "hessian": 1000*y[ii],  # scale like in your plot
                "displaced-atom": n,         # store the atom index
                "atom": np.arange(hessian.shape[1])[ii],  # store the atom index
                "molecule" : molecule[ii], 
                # "delta_F_x": 1000*delta_F[k, ii, 0],
                # "delta_F_y": 1000*delta_F[k, ii, 1],
                # "delta_F_z": 1000*delta_F[k, ii, 2],
            })
            # Append to main DataFrame
            df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    df["tag"] = args.tag

    # Save to a text file
    print(f"\n\tSaving figure to file '{args.output}' ... ",end="")
    df.to_csv(args.output, index=False, float_format=float_format)
    print("done")

    #------------------#
    fig,ax = plt.subplots(1,1,figsize=(6,4)) 
    
    for n,atom in enumerate(nonzero_atoms):
        x = distances[atom]
        for k in range(3*n,3*n+3):
            if k >= len(hessian):
                break
            y = hessian[k,:]
            ii = x != 0
            plt.scatter(x[ii],1000*y[ii],color=colors[n],label=symbols[n] if k==3*n else None,s=1)
        
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"interatomic distance [$\AA$]")
    plt.ylabel(r"$|\Delta \mathbf{F}|/\Delta \mathbf{R}$ [meV/$\AA^2$]")
    plt.ylim(1e-3,1e5)
    
    max_pbc = max_mic_distance(original.get_cell())
    a = original.get_cell().cellpar()[:2].min()

    plt.vlines(max_pbc,1e-3,1e5,ls="--",color="purple",lw=1,label="max. PBC")
    plt.vlines(a,1e-3,1e5,ls="--",color="green",lw=1,label=r"min |$\mathbf{a}_{\alpha}$|")
    
    plt.legend(**legend_options,loc="lower left")
    plt.grid()
    plt.tight_layout()
    
    print(f"\tSaving figure to file '{args.plot}' ... ",end="")
    plt.savefig(args.plot,dpi=300)
    print("done")
        
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()


