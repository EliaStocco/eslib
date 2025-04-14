#!/usr/bin/env pythons
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functions import unique_elements


#---------------------------------------#
# Description of the script's purpose
description = "Divide a supercell into unitcell."

# ToDo:
# This script works only on specific cases:
# - all the atoms in the unitcells have scaled coordinates < 1 (along all components)
# - the supercell is of the form N x N x N

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"          , **argv, required=True , type=str  , help="input file")
    parser.add_argument("-if", "--input_format"   , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--number_of_cells", **argv, required=True , type=int  , help="number of unitcells in the supercell along each direction")
    parser.add_argument("-o" , "--output"         , **argv, required=False, type=str  , help="*.txt output file with index of the unitcell (default: %(default)s)", default="unit-cells.txt")
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
    print("\tFinding unit cells ... ", end="")
    pos = atoms.get_scaled_positions()
    pos = np.asarray(( pos * args.number_of_cells ) // 1).astype(int)

    output = np.zeros((len(pos),4),dtype=int)
    output[:,1:4] = pos
    
    pos_tuples = [tuple(p) for p in pos]
    uelem, indices, inverse_indices = unique_elements(pos_tuples)
    output[:,0] = np.asarray(indices,dtype=int)
    
    print("done")
    
    assert len(uelem) == np.power(args.number_of_cells,3), \
        f"Number of unique unit-cells ({len(uelem)}) does not match the expected number ({np.power(args.number_of_cells,3)})"
    assert len(set([len(a) for a in inverse_indices ])) == 1, \
        f"Number of atoms in each unit-cell is not the same ({set([len(a) for a in inverse_indices ])})"

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