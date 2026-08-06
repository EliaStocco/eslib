#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from ase import Atoms
from typing import List
from eslib.formatting import esfmt
from eslib.input import str2bool

#---------------------------------------#
description = "Create a path bridging two atomic structures, interpolating both positions and unit cells. Useful for variable-cell NEB calculations."

#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-a", "--structure_A", type=str, **argv, help="atomic structure A")
    parser.add_argument("-b", "--structure_B", type=str, **argv, help="atomic structure B")
    parser.add_argument("-n", "--number", type=int, **argv, help="number of inner structures")
    parser.add_argument("-e", "--external", type=float, **argv, help="external structures [between 0 and 1] (default: %(default)s)", default=0)
    parser.add_argument("-f", "--fix_com", type=str2bool, **argv, help="fix the center of mass (default: %(default)s)", default=False)
    parser.add_argument("-N", "--index_com", type=int, **argv, help="index of the structure whose center of mass will be considered (default: %(default)s)", default=0)
    parser.add_argument("-c", "--cartesian", type=str2bool, **argv, help="use Cartesian coordinates (default: %(default)s)", default=False)
    parser.add_argument("-o", "--output", type=str, **argv, help="output file with the path", default="path.extxyz")
    return parser

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #-------------------#
    print("\tReading atomic structure A from file '{:s}' ... ".format(args.structure_A), end="")
    A = read(args.structure_A, index=0)
    print("done")

    #-------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B, index=0)
    print("done")

    #-------------------#
    if len(A) != len(B) or not np.array_equal(A.numbers, B.numbers):
        raise ValueError("The two structures must contain the same atoms in the same order.")
    if not np.array_equal(A.pbc, B.pbc):
        raise ValueError("The two structures must have the same periodic boundary conditions.")

    #-------------------#
    N = args.number
    T = np.linspace(0 - args.external, 1 + args.external, N + 2)
    pathcell = np.empty((N + 2, 3, 3))
    Acell = np.asarray(A.cell)
    Bcell = np.asarray(B.cell)
    for n, t in enumerate(T):
        pathcell[n] = Acell * (1 - t) + t * Bcell

    #-------------------#
    pathpos = np.zeros((N + 2, *A.positions.shape))
    if args.cartesian:
        print("\n\tComputing the path positions (Cartesian) and cells ... ", end="")
        Apos = A.get_positions()
        Bpos = B.get_positions()
        for n, t in enumerate(T):
            pathpos[n] = Apos * (1 - t) + t * Bpos
    else:
        print("\n\tComputing the path positions (Scaled) and cells ... ", end="")
        As = A.get_scaled_positions()
        Bs = B.get_scaled_positions()
        Bs[(As - Bs) > +0.5] += 1
        Bs[(As - Bs) < -0.5] -= 1
        for n, t in enumerate(T):
            pathpos[n] = As * (1 - t) + t * Bs
    print("done")

    #-------------------#
    N = pathpos.shape[0]
    print("\tn. of structures in the path: '{:d}'".format(N))

    #-------------------#
    print("\tCreating the path ... ", end="")
    path: List[Atoms] = [None] * N
    for n in range(N):
        path[n] = A.copy()
        # Set the cell first: in scaled mode the coordinates below belong to it.
        path[n].set_cell(pathcell[n])
        if args.cartesian:
            path[n].set_positions(pathpos[n])
        else:
            path[n].set_scaled_positions(pathpos[n])
    print("done")

    #-------------------#
    if args.fix_com:
        print("\tFixing the center of mass ... ", end="")
        reference: Atoms = path[args.index_com]
        com = reference.get_center_of_mass()
        for n in range(N):
            path[n].set_center_of_mass(com, scaled=False)
        print("done")

    #-------------------#
    print("\n\tSaving the path to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=path, filename=args.output)
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()
