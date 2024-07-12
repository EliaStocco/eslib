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
from eslib.classes.append import AppendableArray

#---------------------------------------#
# Description of the script's purpose
description = "Compute the interatomic distances for each snapshot."
# documentation = "This script is mainly targeted for bulk water systems."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str, help="*.txt output file with the indices of the outliers (default: %(default)s)", default=None)
    # parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str  , help="molecule name (default: %(default)s)", default="molecule")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tNumber atomic structures: ",len(trajectory))

    #------------------#
    print("\tChecking interatomic distances:")
    Nt = len(trajectory)
    accumulate_distances = AppendableArray()
    for n,atoms in enumerate(trajectory):
        # print("\t - atomic structure {:d}/{:d}".format(n+1,Nt),end="\r")

        distances = atoms.get_all_distances(mic=True,vector=False)
        Natoms = len(atoms)
        assert distances.shape == (Natoms,Natoms)
        assert np.allclose(distances,distances.T)
        assert np.allclose(np.diagonal(distances),0)

        all_dists = (distances[np.triu_indices(Natoms, 1)]).flatten()
        assert len(all_dists) == Natoms*(Natoms-1)//2
        del distances

        print("\tmin: {:f} A, max: {:f} A, mean: {:f} A, std: {:f} A".format(np.min(all_dists),np.max(all_dists),np.mean(all_dists),np.std(all_dists)))

        # accumulate_distances.append(all_dists)

        # distances_matrix = np.zeros((Natoms, Natoms))
        # np.fill_diagonal(distances_matrix, 0)
        # distances_matrix[np.triu_indices(Natoms, 1)] = all_dists
        # distances_matrix += distances_matrix.T
        # assert np.allclose(distances_matrix,distances)

    # accumulate_distances = accumulate_distances.finalize()

    print("\n\n\tWell done, everything okay.")

#---------------------------------------#
if __name__ == "__main__":
    main()



