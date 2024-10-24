#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
import argparse

import numpy as np
from ase.io import read

from eslib.formatting import esfmt
from eslib.show import matrix2str
from eslib.tools import find_transformation

#---------------------------------------#
description     = "Compute the trasformation matrix M(A->B) between the lattice vector of the atomic configurations A and B."

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a" , "--structure_A"  , type=str, **argv, help="atomic structure A [cell]")
    parser.add_argument("-b" , "--structure_B"  , type=str, **argv, help="atomic structure B [supercell]")
    parser.add_argument("-o" , "--output"       , type=str, **argv, help="output file for the trasformatio matrix", default=None)
    parser.add_argument("-of", "--output_format", type=str, **argv, help="output format for np.savetxt (default: %(default)s)", default='%24.18e')
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #-------------------#
    # structure A
    print("\tReading structure A from input file '{:s}' ... ".format(args.structure_A), end="")
    A = read(args.structure_A)
    print("done")

    print("\tCell A:")
    cell = np.asarray(A.cell).T
    line = matrix2str(cell.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    # structure B
    print("\tReading structure B from input file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B)
    print("done")

    print("\tCell B:")
    cell = np.asarray(B.cell).T
    line = matrix2str(cell.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    # trasformation
    size,M = find_transformation(A,B)
    print("\tTrasformation matrix M(A->B):")
    line = matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    det = np.linalg.det(M)
    print("\tdet(M): {:6.4f}".format(det))

    #-------------------#
    if args.output is not None:
        print("\n\tSaving transformation matrix to file '{:s}' ... ".format(args.output),end="")
        np.savetxt(args.output,M,fmt=args.output_format)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()