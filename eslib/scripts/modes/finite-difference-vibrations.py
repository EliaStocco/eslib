#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, warning, float_format, error
from eslib.classes.normal_modes import NormalModes
import numpy as np
from typing import List, Tuple

FNAME = "forces"

#---------------------------------------#
# Description of the script's purpose
description = "Compute the normal modes given the forces of the displaced structures along the cartesian axis."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-p" , "--positions"       , **argv, required=True , type=str  , help="xyz/extxyz file with the displaced atomic structures [au] (default: %(default)s)", default='replay.xyz')
    parser.add_argument("-f" , "--forces"          , **argv, required=False, type=str  , help="xyz/extxyz file with the forces (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--name"            , **argv, required=False, type=str  , help="name of the forces array in the positions file (default: %(default)s)", default=None)
    parser.add_argument("-pf", "--positions_format", **argv, required=False, type=str  , help="positions file format (default: %(default)s)", default='ipi')
    parser.add_argument("-ff", "--forces_format"   , **argv, required=False, type=str  , help="forces file format (default: %(default)s)", default='ipi')
    parser.add_argument("-pu", "--positions_unit"  , **argv, required=False, type=str  , help="positions unit (default: %(default)s)", default='atomic_unit')
    parser.add_argument("-fu", "--forces_unit"     , **argv, required=False, type=str  , help="forces unit (default: %(default)s)", default='atomic_unit')
    parser.add_argument("-s" , "--displacement"    , **argv, required=True , type=float, help="displacement")
    parser.add_argument("-o" , "--output"          , **argv, required=False, type=str  , help="pickle output file with the vibrations (default: %(default)s)", default='vibrations.pickle')
    return parser

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
    print("\tReading the displaced atomic structures from file '{:s}' ... ".format(args.positions), end="")
    atoms:AtomicStructures = AtomicStructures.from_file(file=args.positions,format=args.positions_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(atoms)))
    # N = len(atoms)
    # if N % 2 == 1:
    #     print("\t{:s}: you provided an odd number of structures, then the first one will be discarded.".format(warning))
    #     atoms = atoms[1:]

    #------------------#
    if args.forces is None and args.name is None:
        print("\t{:s}: please provide -f,--forces or -n,--name. They cannot be both None.".format(error))
        return -1
    elif args.forces is not None and args.name is not None:
        print("\t{:s}: please provide -f,--forces or -n,--name. They cannot be both provided.".format(error))
        return -1
    elif args.forces is not None:
        print("\tReading the forces from file '{:s}' ... ".format(args.forces), end="")
        forces:AtomicStructures = AtomicStructures.from_file(file=args.forces,format=args.forces_format)
        print("done")
        data = forces.get_array("positions")
        atoms.set_array("forces",data)
    elif args.name is not None:
        if not atoms.is_there(args.name):
            print("\t{:s}: '{:s}' is not provided in file '{:s}'.".format(error,args.name,args.positions))
            return -1
        forces = atoms.get(args.name,what="arrays")
        atoms.set(FNAME,forces,what="arrays")

    #------------------#
    # unit
    if args.positions_unit is not None and args.positions_unit not in ["au","atomic_unit"]:
        print("\tConverting positions from '{:s}' to '{:s}'.".format(args.positions_unit,"atomic_unit"))
        atoms.convert(name="positions",family="length",_from=args.positions_unit,_to="atomic_unit",inplace=True)
        print("done")
    if args.forces_unit is not None and args.forces_unit not in ["au","atomic_unit"]:
        print("\tConverting forces from '{:s}' to '{:s}'.".format(args.forces_unit,"atomic_unit"))
        atoms.convert(name=FNAME,family="force",_from=args.forces_unit,_to="atomic_unit",inplace=True)
        print("done")
    
    #------------------#
    N = len(atoms)
    if (N-1) % 2 == 0:
        print("\tProvided an odd number of structures: the first one will be discarded.")
        atoms = atoms.subsample("1:")
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
    print("\tComputing the force constant matrix ... ",end="")
    first = atoms[0]
    Ndof = first.get_global_number_of_atoms()*3
    force_constants = np.full((Ndof,Ndof),np.nan)
    forces = atoms.get(FNAME,what="arrays")
    for n in range(Ndof):
        # The order must respect the one in 'cartesian-displacements.py'
        force_constants[n,:] = - np.asarray( ( forces[2*n,:] - forces[2*n+1,:] ) / (2*displacement) ).flatten()

    nm = NormalModes(Nmodes=Ndof,Ndof=Ndof,ref=first)
    nm.set_force_constants(force_constants)
    print("done")

    #------------------#
    print("\tDiagonalizing the dynamical matrix ... ",end="")
    nm.diagonalize()
    print("done")
    
    #------------------#
    print("\n\tWriting the normal modes to file '{:s}' ... ".format(args.output), end="")
    nm.to_pickle(args.output)
    print("done")
   
#---------------------------------------#
if __name__ == "__main__":
    main()