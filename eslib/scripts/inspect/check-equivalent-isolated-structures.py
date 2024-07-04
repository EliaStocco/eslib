#!/usr/bin/env python
from ase.io import read
from eslib.formatting import esfmt, warning
from ase import Atoms
import numpy as np
from ase.io import read, write
from eslib.tools import cart2frac, frac2cart
from eslib.input import slist
from typing import List
from eslib.classes.trajectory import AtomicStructures
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Checks that the intramolecular distances are the same despite the pbc."
documentation = "This script is mainly targeted for bulk water systems."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str  , help="molecule name (default: %(default)s)", default="molecule")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tNumber atomic structures: ",len(trajectory))

    #------------------#
    print("\tChecking interatomic distances:")
    Nt = len(trajectory)
    for n,atoms in enumerate(trajectory):
        print("\t - atomic structure {:d}/{:d}".format(n+1,Nt),end="\r")
        
        molecules = atoms.arrays[args.molecule]
        for molecule in np.unique(molecules):
            indices = [ n for n,a in enumerate(molecules) if a == molecule]
            N = len(indices)
            pbc_distances = np.zeros((N,N))
            nopbc_distances = np.zeros((N,N))
            for ii in range(N):
                pbc_distances[ii,:]   = atoms.get_distances(indices[ii],indices,mic=True,vector=False)
                nopbc_distances[ii,:] = atoms.get_distances(indices[ii],indices,mic=False,vector=False)

            assert np.allclose(pbc_distances,nopbc_distances)

    print("\n\n\tWell done, everything okay.")

#---------------------------------------#
if __name__ == "__main__":
    main()



