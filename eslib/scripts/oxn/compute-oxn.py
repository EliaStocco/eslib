#!/usr/bin/env python
import numpy as np
from ase.io import write
from eslib.classes.trajectory import AtomicStructures# , info, array
from eslib.formatting import esfmt
from eslib.physics import compute_dipole_quanta
import matplotlib.pyplot as plt
from eslib.tools import frac2cart
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Save an 'array' or 'info' from an extxyz file to a txt file."


#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="input file [extxyz]")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-f" , "--fix"     , **argv, required=False, type=str2bool, help="fix branch", default=True)
    parser.add_argument("-k" , "--keyword"     , **argv, required=False, type=str, help="keyword the dipoles (default: 'dipole')", default="dipole")
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output file (default: %(default)s)" , default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #---------------------------------------#
    Rstart:np.ndarray = atoms[0].get_positions()# .flatten()
    Rend:np.ndarray = atoms[-1].get_positions()# .flatten()
    R = Rend - Rstart

    R2 = np.linalg.norm(R,axis=1)**2
    non_zero_indices = R2 != 0

    symbols = np.asarray(atoms[0].get_chemical_symbols())[non_zero_indices]
    Natoms = atoms[0].get_global_number_of_atoms()
    print("\n\tIndices of displaced atoms: ",np.arange(Natoms)[non_zero_indices].astype(int))
    print("\tSymbols of displaced atoms: ",symbols)
    assert all(elem == symbols[0] for elem in symbols) , "You displaced atoms of different species."
    
    #---------------------------------------#
    # dipole = atoms.get(args.keyword)

    quanta:np.ndarray= compute_dipole_quanta(atoms,args.keyword)[1]
    shift = np.floor(quanta[0,:])
    quanta -= shift
    # assert quanta.shape == dipole.shape

    # quanta *= np.sign(R_filtered[0,:]) # just to have the correct slope in the plot

    if args.fix:

        print("\n\tFixing branch ...  ",end="")

        def fix_quanta(quanta):

            if quanta.ndim == 1:
                quanta = quanta.reshape((-1,1))

            quanta = np.unwrap(quanta,period=1,axis=0).astype(float)

            dQ = np.diff(quanta,axis=0)
            # dQ = np.unwrap(dQ,axis=0,period=1)
            
            # add = np.zeros_like(quanta)
            # add[1:,:] = np.cumsum(dQ, axis=0).astype(float)
            # new_quanta = add + quanta[0,:]

            return quanta, dQ

        new_quanta, dQ = fix_quanta(quanta)

        threshold = 3
        array = dQ
        mean = np.mean(array,axis=0)
        std = np.std(array,axis=0)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outliers = (array < lower_bound) | (array > upper_bound)

        # Step 1: Create new_outliers array with an additional row for the first element
        new_outliers = np.zeros((outliers.shape[0] + 1, 3), dtype=bool)
        new_outliers[0, :] = False  # Mark the first element as okay
        new_outliers[1:] = outliers  # Copy the rest of the outliers

        for n in range(3):
            # Step 2: Extract non-outlier elements from new_quanta
            tmp = new_quanta[~new_outliers[:,n],n]# .reshape((-1, 3))
            # tmp = tmp.reshape((-1, 3))

            # Step 3: Fix the quanta for the non-outlier elements
            new_new_quanta, dQ = fix_quanta(tmp)

            # Step 4: Update new_quanta with fixed values and mark outliers as NaN
            new_quanta[new_outliers] = np.nan
        
            new_quanta[~new_outliers[:,n],n] = new_new_quanta.flatten()# [:,n]

        quanta = new_quanta.copy()

        print("done")

    dipole = frac2cart(atoms[0].get_cell(),quanta)

    Dstart = dipole[0]
    Dend = dipole[-1]
    DeltaD = (Dend-Dstart)# /atoms[0].get_volume()
    # Filter R2 and R using the non-zero indices
    R2_filtered = R2[non_zero_indices]
    R_filtered = R[non_zero_indices]

    # Use the filtered R2 and R for further calculations
    N = np.divide(DeltaD @ R_filtered.T, R2_filtered)
    assert np.std(N) < 1e-3

    DeltaN = np.mean(N)
    N = DeltaN / len(N)
    print("\n\tOxidation number for '{:s}': {:f}".format(symbols[0],np.round(N,1)))

    if args.output is not None:

        print("\n\tCreating plot ...  ",end="")
        quanta *= np.sign(R_filtered[0,:]) # just to have the correct slope in the plot
        
        label = ["x","y","z"]
        fig, ax = plt.subplots(figsize=(4, 3))
        for n in range(3):
            y = quanta[:,n]
            # Filter out NaN values
            mask = ~np.isnan(y)
            x = np.linspace(0,1,len(quanta),endpoint=True)
            x = x[mask]
            y = y[mask]
            ax.plot(x,y,label=label[n],marker='.')
        ax.grid(True)
        ax.set_xlabel("path")
        ax.set_ylabel("dipole/quantum [unitless]")
        ax.set_yticks(np.arange(np.floor(np.nanmin(quanta)), np.ceil(np.nanmax(quanta)) + 1, 1))
        ax.legend(facecolor='white', framealpha=1,edgecolor="black")
        plt.tight_layout()
        print("done")

        print("\tSaving plot to file '{:s}' ... ".format(args.output), end="")
        plt.savefig(args.output)
        print("done")
        
        
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()

