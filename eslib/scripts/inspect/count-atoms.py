#!/usr/bin/env python
from typing import List

import numpy as np
from ase import Atoms
from ase.io import read, write

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.input import slist, str2bool
from eslib.tools import cart2frac, frac2cart

#---------------------------------------#
# Description of the script's purpose
description = "Count the atoms in a structure for each chemical species."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input, format=args.input_format,index=0)[0]
    print("done")
    N = atoms.get_global_number_of_atoms()
    print("\tNumber of atoms: ",N)
    
    symbols = atoms.get_chemical_symbols()
    species = np.unique(symbols)

    for s in species:
        n = sum(1 for a in symbols if a == s)
        print(f"\t {s}: {n} ({100*n/N:.1f}%)")

#---------------------------------------#
if __name__ == "__main__":
    main()



