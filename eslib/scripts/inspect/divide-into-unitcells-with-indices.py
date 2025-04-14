#!/usr/bin/env pythons
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functions import invert_indices, unique_elements

#---------------------------------------#
# Description of the script's purpose
description = "Divide a supercell into unitcell using the indices produced by 'map-atoms.py'."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="input file")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--supercell_size", **argv, required=True , type=int , help="number of unitcells in the supercell along each direction")
    parser.add_argument("-n" , "--indices"       , **argv, required=True , type=str  , help="file with indices of the unitcell")
    parser.add_argument("-p" , "--plot"          , **argv, required=False, type=str  , help="foldr for the plots (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str  , help="*.txt output file with index of the unitcell (default: %(default)s)", default="unit-cells.txt")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",atoms.get_global_number_of_atoms(),end="\n\n")

    #------------------#
    print("\tReading indices of the unitcell from file '{:s}' ... ".format(args.indices), end="")
    indices = np.loadtxt(args.indices, dtype=int)
    print("done")
    
    #------------------#
    print("\tDividing the supercell into unitcells ... ", end="")
    inv_indices = invert_indices(indices)
    atoms = atoms[indices]
    N = np.power(args.supercell_size,3)
    unit_cell_index = np.tile(np.arange(N),len(atoms)//N)
    assert len(unit_cell_index) == len(atoms), \
        f"Number of atoms ({len(atoms)}) does not match the expected number ({len(unit_cell_index)})"
    unit_cell_index = unit_cell_index[inv_indices]
    atoms = atoms[inv_indices]
    print("done")
    
    #------------------#
    if args.plot is not None:
        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt
        # for i in range(N):
        #     tmp = atoms[i::N]       
            
        #     fig, ax = plt.subplots()
        #     plot_atoms(tmp, ax)  # Optional rotation
        #     plt.savefig(f"{args.plot}/unitcell.n={i}.png") 
        #     plt.close(fig)
            
        
        for i in range(N):
            tmp = atoms[unit_cell_index==i]
            
            fig, ax = plt.subplots()
            plot_atoms(tmp, ax)  # Optional rotation
            plt.savefig(f"{args.plot}/unitcell.n={i}.png") 
            plt.close(fig)
    
    #------------------#
    print("\tPreparing output ... ", end="")
    uelem, indices, inv_indices = unique_elements(unit_cell_index)
    assert len(uelem) == np.power(args.supercell_size,3), \
        f"Number of unique unit-cells ({len(uelem)}) does not match the expected number ({np.power(args.supercell_size,3)})"
    assert len(set([len(a) for a in inv_indices ])) == 1, \
        f"Number of atoms in each unit-cell is not the same ({set([len(a) for a in indices ])})"
    output = np.zeros((len(atoms),4),dtype=int)
    output[:,0] = unit_cell_index
    
    pos = atoms.get_scaled_positions()
    pos = np.asarray(( pos * args.supercell_size ) // 1).astype(int)
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