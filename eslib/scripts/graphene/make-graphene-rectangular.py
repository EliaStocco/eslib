#!/usr/bin/env python
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format

#---------------------------------------#
description = "Create a rectangular supercell for graphene."

documentation = "https://mattermodeling.stackexchange.com/questions/11042/how-can-visualize-a-rectangular-super-cell-of-graphene-by-vesta"

#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, type=str, help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"       , **argv, type=str, help="output file (default: %(default)s)", default="matrix.txt")
    return  parser

#---------------------------------------#
@esfmt(prepare_parser,description,documentation)
def main(args):

    #-------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    cellpar = atoms.cell.cellpar()
    assert np.allclose(cellpar[0],cellpar[1]), "a!=b"
    assert np.allclose(cellpar[3],cellpar[4]), "alpha!=beta"
    
    Cell = np.asarray(atoms.cell)
    cell = np.asarray(Cell[:2,:2] / cellpar[0])
    
    case_a = np.asarray([1.,0,0.5,np.sqrt(3.)/2.])
    case_b = np.asarray([1.,0.5,0,np.sqrt(3.)/2.])
    
    TOLERANCE = 1e-4
    if np.allclose(case_a,cell.flatten(),atol=TOLERANCE):
        matrix = np.asarray([2,0,-1,2])
    elif np.allclose(case_b,cell.flatten(),atol=TOLERANCE):
        matrix = np.asarray([2,1,0,2])
    else:
        raise ValueError("What cell did you provide?")
    
    Matrix = np.zeros((3,3),dtype=float)
    Matrix[:2,:2] = matrix.reshape((2,2))
    Matrix[2,2] = 1
    
    print(f"\tSaving matrix to file '{args.output}' ... ")
    np.savetxt(args.output,Matrix,fmt=float_format)
    print("\tdone")
    
#---------------------------------------#
if __name__ == "__main__":
    main()
