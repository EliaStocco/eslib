#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, warning, float_format
from eslib.tools import convert
import numpy as np
from typing import List, Tuple

#---------------------------------------#
# Description of the script's purpose
description = "Compute the BEC tensors given the dipoles of the displaced structures along the cartesian axis."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=False, type=str  , help="input file with the displaced atomic structures [au] (default: 'replay.xyz')", default='replay.xyz')
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-d" , "--dipoles"     , **argv, required=True , type=str  , help="file with the dipoles")
    parser.add_argument("-s" , "--displacement", **argv, required=True , type=float, help="displacement")
    parser.add_argument("-u" , "--unit"        , **argv, required=False, type=str  , help="dipole unit (default: 'atomic_unit')", default='atomic_unit')
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output file with the BEC tensors (default: 'bec.txt')", default='bec.txt')
    return parser# .parse_args()

#---------------------------------------#
def split_list(lst:List[Atoms])->Tuple[List[Atoms],List[Atoms]]:
    left = list()
    right = list()
    for i in range(len(lst)):
        if i % 2 == 0:
            left.append(lst[i])
        else:
            right.append(lst[i])
    return left, right

#---------------------------------------#
def check_off_diagonal(matrix:np.ndarray)->bool:
    rows = len(matrix)
    cols = len(matrix[0])
    
    for i in range(rows):
        for j in range(cols):
            if i != j and matrix[i][j] != 0:
                return False
    return True
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the displaced atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:List[Atoms] = list(AtomicStructures.from_file(file=args.input,format=args.input_format,index=":"))
    print("done")

    N = len(atoms)
    if N % 2 == 1:
        print("\t{:s}: you provided an odd number of structures, then the first one will be discarded.".format(warning))
        atoms = atoms[1:]

    #------------------#
    # dipoles
    print("\tReading dipoles from file '{:s}' ... ".format(args.dipoles), end="")
    dipoles = np.loadtxt(args.dipoles)
    print("done")

    if dipoles.ndim != 2:
        raise ValueError("'dipoles' should have dimension 2.")
    if dipoles.shape[0] % 2 == 1:
        print("\t{:s}: you provided an odd number of dipoles, then the first one will be discarded.".format(warning))
        dipoles = dipoles[1:]

    #------------------#
    # unit
    if args.unit not in ["au","atomic_unit"]:
        factor = convert(1,"electric-dipole",args.unit,"atomic_unit")
        print("\tConverting the dipoles from file '{:s}' to 'atomic_unit'.".format(args.unit))
        print("\tMultiplication factor: ",factor)
        dipoles *= factor
    
    #------------------#
    N = len(atoms)
    if N % 2 == 1:
        raise ValueError("The number of provided structures has to be even.")
    if N%6 != 0 :
        raise ValueError("The number of provided structures has to be a multiple of 6.")
    
    #------------------#
    # displacement
    left, right = split_list(atoms)
    left  = np.asarray([ a.get_positions().flatten() for a in left  ])
    right = np.asarray([ a.get_positions().flatten() for a in right ])

    # displacements = np.absolute(left-right).diagonal()
    # if not np.allclose(displacements,displacements[0]):
    #     raise ValueError("The displacements should be all the same.")
    # if not check_off_diagonal(left-right):
    #     raise ValueError("coding error: off diagonal displacements")

    # they are all the same
    # displacement = abs(displacements[0]/2.)
    displacement = args.displacement
    print("\tThe provided structures have all been displaced by {:f} atomic unit".format(displacement))

    #------------------#
    N_2 = int(N/2)
    BEC = np.full((N_2,3),np.nan)
    for n in range(N_2):
        # The order must respect the one in 'cartesian-displacements.py'
        BEC[n,:] = ( dipoles[2*n,:] - dipoles[2*n+1,:] ) / (2*displacement)
    
    #------------------#
    print("\n\tWriting the BEC tensors to file '{:s}' ... ".format(args.output), end="")
    try:
        np.savetxt(args.output,BEC,fmt=float_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    
#---------------------------------------#
if __name__ == "__main__":
    main()