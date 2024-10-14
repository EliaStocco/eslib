#!/usr/bin/env python
import numpy as np
import os
from ase.build import make_supercell
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from phonopy.structure.cells import get_supercell
from eslib.tools import ase2phonopy, phonopy2ase

#---------------------------------------#
description = "Create a supercell for the given atomic structures."

documentation = \
"""
-t/--type: 
    str (ASE default: "cell-major")
    how to order the atoms in the supercell

    "phonopy":
    something compatible with phonopy
    
    "phonopy-smith"
    something compatible with phonopy and Smith algorithm
    
    "cell-major":
    [atom1_shift1, atom2_shift1, ..., atom1_shift2, atom2_shift2, ...]
    i.e. run first over all the atoms in cell1 and then move to cell2.

    "atom-major":
    [atom1_shift1, atom1_shift2, ..., atom2_shift1, atom2_shift2, ...]
    i.e. run first over atom1 in all the cells and then move to atom2.
    This may be the order preferred by most VASP users.
"""
choices = ['phonopy', 'phonopy-smith', 'cell-major', 'atom-major']

#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, type=str, help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--matrix"       , **argv, type=str, help="txt file with the 3x3 transformation matrix")
    parser.add_argument("-t" , "--type"         , **argv, type=str, help=f"order type {choices}"+" (default: %(default)s)" , default="phonopy",choices=choices)
    parser.add_argument("-o" , "--output"       , **argv, type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str, help="output file format (default: %(default)s)", default=None)
    return  parser

#---------------------------------------#
@esfmt(prepare_parser,description,documentation)
def main(args):

    #-------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #-------------------#
    print("\tReading transformation matrix from file '{:s}' ... ".format(args.matrix), end="")
    if os.path.isfile(args.matrix):
        matrix = np.loadtxt(args.matrix,dtype=int)
    else:
        try:
            matrix = np.array([int(x) for x in str(args.matrix).split()])
            if matrix.flatten().shape == (9,):
                matrix = matrix.reshape(3,3)
            elif matrix.flatten().shape == (3,):
                matrix = np.diag(matrix)
            else:
                raise ValueError("The argument after '--matrix' should be a file or a string of 3/9 integers separated by spaces.")
        except ValueError:
            raise ValueError("The argument after '--matrix' should be a file or a string of 3/9 integers separated by spaces.")
    print("done")

    #-------------------#
    print("\n\tCreating the supercells ... ", end="")
    supercell = [None] * len(structures)
    for n,atoms in enumerate(structures):
        if args.type in ['cell-major', 'atom-major']:
            supercell[n] = make_supercell(atoms,matrix,wrap=False,order=args.type)
        else:
            atoms = ase2phonopy(atoms)
            is_old_style = False if args.type == 'phonopy-smith' else True
            supercell[n] = get_supercell(atoms,supercell_matrix=matrix,is_old_style=is_old_style)
            supercell[n] = phonopy2ase(supercell[n])
    supercell = AtomicStructures(supercell)
    print("done")

    #-------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    supercell.to_file(file=args.output,format=args.output_format)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()
