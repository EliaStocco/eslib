#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import argparse
import numpy as np
from ase.io import read
from ase.geometry import get_distances
from eslib.show import matrix2str
from eslib.tools import find_transformation
from eslib.input import str2bool
from eslib.tools import sort_atoms
from eslib.formatting import esfmt

#---------------------------------------#
description     = "Compute the difference between two atomic structures. "

#---------------------------------------#
def prepare_parser(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a", "--structure_A",  type=str    ,**argv,help="atomic structure A [au]")
    parser.add_argument("-b", "--structure_B",  type=str     ,**argv,help="atomic structure B [au]")
    parser.add_argument("-s", "--sort"       ,  type=str2bool,**argv,help="whether to sort the second structure (dafault: false)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #-------------------#
    print("\tReading atomic structure A from file '{:s}' ... ".format(args.structure_A), end="")
    A = read(args.structure_A)
    print("done")

    #-------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B)
    print("done")

    #-------------------#
    # sort
    if args.sort:
        print("\n\tSorting the atoms of the second structure  ... ", end="")
        B, indices = sort_atoms(A, B)
        print("done")

    #-------------------#
    # cells
    pbc = np.all(A.get_pbc()) and np.all(B.get_pbc())
    if pbc:
        _ , M = find_transformation(A,B)
        print("\n\tTrasformation matrix M(A->B):")
        line = matrix2str(M,digits=2,col_names=["1","2","3"],cols_align="^",width=6)
        print(line)

    #---------------------------------------#
    # positions
    print("\n\tPositions differences (cartesian, norm, and fractional):")
    # get_distances(A,B,mic=True)
    # diff = A.positions - B.positions
    a,b = get_distances(A.positions,B.positions,cell=A.get_cell(),pbc=A.get_pbc())
    diff = np.asarray([ a[i,i,:] for i in range(len(a))])
    d = np.diag(b)
    # np.allclose(np.linalg.norm(a,axis=2),b)
    # from icecream import ic
    # ic(d.shape)
    # M = np.concatenate([diff,np.linalg.norm(diff,axis=1)[:, np.newaxis]], axis=1)
    M = np.concatenate([diff,d[:, np.newaxis]], axis=1)
    col_names=["Rx","Ry","Rz","norm"]
    if pbc:
        fractional = ( np.linalg.inv(A.get_cell().T) @ diff.T ).T
        M = np.concatenate([M,fractional], axis=1)
        col_names += ["fx","fy","fz"]
        #M = np.concatenate(A.positions - B.positions],
    line = matrix2str(M,digits=3,col_names=col_names,cols_align="^",width=8,row_names=A.get_chemical_symbols())
    print(line)

    return 0

if __name__ == "__main__":
    main()