#!/usr/bin/env python

import numpy as np
import pandas as pd
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt


#---------------------------------------#
# Description of the script's purpose
description = "Merge two dataset (the second into the first)."
documentation = \
"This script is based on 'match-snapshots.py':\n" +\
" - it creates a dataset with all 'unique' snapshots from the provided ones\n" + \
" - for a shared structures it adds on top (replaces) the info and array of the second dataset to the first" 

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a" , "--dataset_A"       , **argv, required=True , type=str  , help="dataset A")
    parser.add_argument("-af", "--dataset_A_format", **argv, required=False, type=str  , help="dataset A format (default: %(default)s)", default=None)
    parser.add_argument("-b" , "--dataset_B"       , **argv, required=True , type=str  , help="dataset B")
    parser.add_argument("-bf", "--dataset_B_format", **argv, required=False, type=str  , help="dataset B format (default: %(default)s)", default=None)
    parser.add_argument("-t" , "--tolerance"       , **argv, required=False, type=float, help="tolerance (default: %(default)s)", default=1e-8)
    parser.add_argument("-od", "--output_dataframe", **argv, required=False, type=str  , help="output dataframe (default: %(default)s)", default="description.csv")
    parser.add_argument("-o" , "--output"          , **argv, required=True , type=str  , help="merged dataset")
    parser.add_argument("-ot", "--output_format"   , **argv, required=False, type=str  , help="merged dataset format (default: %(default)s)", default='extxyz')
    return parser

def merge(atoms_A:Atoms,atoms_B:Atoms)->Atoms:
    atoms = atoms_A.copy()
    for key,value in atoms_B.info.items():
        atoms.info[key] = value
    for key,value in atoms_B.arrays.items():
        atoms.arrays[key] = value
    return atoms

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):
    
    #------------------#
    print("\tReading dataset A from file '{:s}' ... ".format(args.dataset_A), end="")
    dataset_A = AtomicStructures.from_file(file=args.dataset_A,format=args.dataset_A_format)
    print("done")
    print("\tNumber of snapshots in dataset A: {:d}".format(len(dataset_A)))
    # shape: (num_structures, num_atoms, 3)
    pos_A = dataset_A.get("positions")
    
    #------------------#
    print("\tReading dataset A from file '{:s}' ... ".format(args.dataset_B), end="")
    dataset_B = AtomicStructures.from_file(file=args.dataset_B,format=args.dataset_B_format)
    print("done")
    print("\tNumber of snapshots in dataset A: {:d}".format(len(dataset_B)))
    # shape: (num_structures, num_atoms, 3)
    pos_B = dataset_B.get("positions")
    
    #------------------#
    all_info_arrays = set()
    for dataset in [dataset_A,dataset_B]:
        for a in dataset:
            for k in a.info.keys():
                all_info_arrays.add(k)
            for k in a.arrays.keys():
                all_info_arrays.add(k)
    all_info_arrays = sorted(all_info_arrays)
    
    #------------------#
    print("\n\tFinding matching snapshots (A->B) ... ", end="")
    indices = np.zeros((len(pos_A),),dtype=int)
    for n,pos in enumerate(pos_A):
        # shape: (num_ref, num_atoms, 3)
        diff = pos_B - pos[None,:,:]
        dist = np.linalg.norm(diff,axis=(1,2))
        idx = np.argmin(dist)  
        
        matches = np.where(dist < args.tolerance)[0]
        if len(matches) > 1:
            print(
                "\n\tWARNING: structure {:d} matches multiple structures "
                "in dataset B: {}".format(
                    n,
                    matches.tolist()
                )
            )
    
        indices[n] = idx  if dist[idx] < args.tolerance else -1
    print("done")
    
    #------------------#
    rows = [None]*len(dataset_A)
    merged = [None]*len(dataset_A) # dataset_A.copy()
    for n,i in enumerate(indices):
        atoms_A:Atoms = dataset_A[n]
        if i != -1:
            atoms_B:Atoms = dataset_B[i]
            merged[n] = merge(atoms_A,atoms_B)
            keys_A = (
                set(atoms_A.info.keys()) |
                set(atoms_A.arrays.keys())
            )
            keys_B = (
                set(atoms_B.info.keys()) |
                set(atoms_B.arrays.keys())
            )
        else:
            merged[n] = atoms_A.copy()
            keys_A = (
                set(atoms_A.info.keys()) |
                set(atoms_A.arrays.keys())
            )

            keys_B = set()
            
        # ------------------#
        # Fill provenance dataframe row
        row = {}
        # original indices
        row["index"] = n
        row["index-A"] = n
        row["index-B"] = i
        for key in all_info_arrays:
            # B overwrites A
            if key in keys_B:
                row[key] = "B"
            elif key in keys_A:
                row[key] = "A"
            else:
                row[key] = "-"
        rows[n] = row
            
    #------------------#
    print("\n\tFinding matching snapshots (B->A) ... ", end="")
    indices = np.zeros((len(pos_B),),dtype=int)
    for n,pos in enumerate(pos_B):
        # shape: (num_ref, num_atoms, 3)
        diff = pos_A - pos[None,:,:]
        dist = np.linalg.norm(diff,axis=(1,2))
        idx = np.argmin(dist)  
        matches = np.where(dist < args.tolerance)[0]
        if len(matches) > 1:
            print(
                "\n\tWARNING: structure {:d} matches multiple structures "
                "in dataset B: {}".format(
                    n,
                    matches.tolist()
                )
            )
            
        indices[n] = idx  if dist[idx] < args.tolerance else -1
    print("done")
    
    #------------------#
    k = len(rows)
    for n,i in enumerate(indices):
        if i == -1:
            atoms_B:Atoms = dataset_B[n].copy()

            merged.append(atoms_B)

            keys_B = (
                set(atoms_B.info.keys()) |
                set(atoms_B.arrays.keys())
            )
            row = {}
            row["index"] = k
            row["index-A"] = i
            row["index-B"] = n
            for key in all_info_arrays:
                if key in keys_B:
                    row[key] = "B"
                else:
                    row[key] = "-"
            rows.append(row)
            k += 1
            
    # ------------------#
    all_info_arrays = ["index"] + ["index-A"] + ["index-B"] + all_info_arrays
    df = pd.DataFrame(rows, columns=all_info_arrays)
    print(f"\tWriting dataframe to file '{args.output_dataframe}' ... ",end="")
    with open(args.output_dataframe, "w") as f:

        f.write("# Merged dataset provenance table\n")
        f.write("#\n")
        f.write("# dataset_A : {}\n".format(args.dataset_A))
        f.write("# dataset_B : {}\n".format(args.dataset_B))
        f.write("# tolerance : {}\n".format(args.tolerance))
        f.write("#\n")
        f.write("# Column meaning:\n")
        f.write("#   index_A = index of the structure in dataset A (-1 = not present)\n")
        f.write("#   index_B = index of the structure in dataset B (-1 = not present)\n")
        f.write("#\n")
        f.write("# Provenance labels:\n")
        f.write("#   A = property/array comes from dataset A\n")
        f.write("#   B = property/array comes from dataset B\n")
        f.write("#       (B overwrites A if present in both)\n")
        f.write("#   - = property/array not present for this structure\n")
        f.write("#\n")

        df.to_csv(f, index=False)
    print("done")
        
    n_A = sum(df["index-A"] != -1)
    assert len(dataset_A) == n_A 
    n_B = sum(df["index-B"] != -1)
    assert len(dataset_B) == n_B 
        
    #---------------------------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting merged dataset to file '{:s}' ... ".format(args.output), end="")
    AtomicStructures(merged).to_file(file=args.output, format=args.output_format)
    print("done")
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
