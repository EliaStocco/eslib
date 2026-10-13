#!/usr/bin/env python
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functions import unique_elements
from eslib.input import ilist

#---------------------------------------#
# Description of the script's purpose
description = "Divide a supercell into unit cells according to the ordering used by 'make-supercell.py'."

supercell_methods = ['phonopy', 'phonopy-smith', 'cell-major', 'atom-major']

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="input file")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--supercell_size", **argv, required=True , type=ilist, help="number of unitcells along each lattice vector: one value for a cubic supercell, or three values (e.g. '10 8 6')")
    parser.add_argument("-m" , "--method"        , **argv, required=False, type=str, choices=supercell_methods, default="phonopy", help="atom ordering used by make-supercell.py (default: %(default)s)")
    parser.add_argument("-n" , "--indices"       , **argv, required=False, type=str, default=None, help="canonical atom-major -> input ordering map; required for phonopy-smith")
    parser.add_argument("-p" , "--plot"          , **argv, required=False, type=str  , help="folder for the plots (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str  , help="*.txt output file with index of the unitcell (default: %(default)s)", default="unit-cells.txt")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    supercell_size = np.asarray(args.supercell_size, dtype=int)
    if supercell_size.size == 1:
        supercell_size = np.repeat(supercell_size, 3)
    if supercell_size.shape != (3,) or np.any(supercell_size <= 0):
        raise ValueError("--supercell_size must contain one positive integer or three positive integers")
    n_unit_cells = int(np.prod(supercell_size))

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",atoms.get_global_number_of_atoms(),end="\n\n")

    #------------------#
    #------------------#
    print("\tDividing the supercell into unitcells ... ", end="")
    if len(atoms) % n_unit_cells != 0:
        raise ValueError(
            f"Number of atoms ({len(atoms)}) is not divisible by the number "
            f"of unit cells ({n_unit_cells})"
        )

    atoms_per_unit_cell = len(atoms) // n_unit_cells
    if args.method in ("atom-major", "phonopy"):
        unit_cell_index = np.tile(np.arange(n_unit_cells), atoms_per_unit_cell)
    elif args.method == "cell-major":
        unit_cell_index = np.repeat(np.arange(n_unit_cells), atoms_per_unit_cell)
    else:
        if args.indices is None:
            raise ValueError(
                f"--indices is required for method '{args.method}'. It must map "
                "canonical atom-major indices to indices in the input structure."
            )
        print("\n\tReading indices of the unitcell from file '{:s}' ... ".format(args.indices), end="")
        indices = np.loadtxt(args.indices, dtype=int)
        print("done")
        if indices.ndim != 1 or len(indices) != len(atoms):
            raise ValueError(
                f"Index map must contain exactly {len(atoms)} entries; got {indices.size}"
            )
        if not np.array_equal(np.sort(indices), np.arange(len(atoms))):
            raise ValueError("Index map must be a permutation of the input atom indices")

        canonical_unit_cell_index = np.tile(np.arange(n_unit_cells), atoms_per_unit_cell)
        unit_cell_index = np.empty_like(canonical_unit_cell_index)
        unit_cell_index[indices] = canonical_unit_cell_index
    print("done")
    
    #------------------#
    if args.plot is not None:
        import os
        import matplotlib
        matplotlib.use("Agg", force=True)
        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt
        
        os.makedirs(args.plot, exist_ok=True)

        # Use common limits so that the location of each cell in the full
        # supercell is apparent when comparing the output images.
        x_all = atoms.positions[:, 0]
        y_all = atoms.positions[:, 1]
        xlim = (x_all.min(), x_all.max())
        ylim = (y_all.min(), y_all.max())

        width = len(str(n_unit_cells - 1))
        for i in range(n_unit_cells):
            tmp = atoms[unit_cell_index == i]
            fig, ax = plt.subplots()
            plot_atoms(tmp, ax)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.set_title(f"Unit cell {i}")
            fig.savefig(
                f"{args.plot}/unitcell.n={i:0{width}d}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

    
    #------------------#
    print("\tPreparing output ... ", end="")
    uelem, indices, inv_indices = unique_elements(unit_cell_index)
    assert len(uelem) == n_unit_cells, \
        f"Number of unique unit-cells ({len(uelem)}) does not match the expected number ({n_unit_cells})"
    assert len(set([len(a) for a in inv_indices ])) == 1, \
        f"Number of atoms in each unit-cell is not the same ({set([len(a) for a in indices ])})"
    output = np.zeros((len(atoms),4),dtype=int)
    output[:,0] = unit_cell_index
    
    pos = atoms.get_scaled_positions()
    pos = np.floor(pos * supercell_size).astype(int) % supercell_size
    output[:,1:4] = pos
    
    for n in uelem:
        tmp = output[inv_indices[n],1:4].astype(float)
        if np.linalg.norm(np.std(tmp,axis=0)) > 1e-8:
            val = np.mean(tmp,axis=0)
            val = np.round(val,0).astype(int)
            output[inv_indices[n],1:4] = val
    
    print("done")

    #------------------#
    print("\tSaving results to file '{:s}' ... ".format(args.output), end="")
    header = \
            f"Col 1: global index of the unit-cell\n" +\
            f"Col 2: unit-cell coordinate along the 1st lattice vector\n" +\
            f"Col 3: unit-cell coordinate along the 2nd lattice vector\n"+\
            f"Col 4: unit-cell coordinate along the 3rd lattice vector"
    np.savetxt(args.output, output, fmt="%8d", header=header)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
