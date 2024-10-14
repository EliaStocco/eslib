#!/usr/bin/env python
import numpy as np
from ase.io import write
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import warning, error, esfmt
from eslib.input import flist
from eslib.classes.atoms_selector import AtomSelector
from ase import Atoms
from eslib.tools import cart2frac, frac2cart

#---------------------------------------#
# Description of the script's purpose
description = "Create a path with a sublattice (atom or set of atoms) displaced to its periodic replica."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, type=str  , help="input file [extxyz]")
    parser.add_argument("-a" , "--atoms"         , **argv, type=str  , help="atoms to be displaced")
    parser.add_argument("-n" , "--number"        , **argv, type=int  , help="number of intermediate structures in the path")
    parser.add_argument("-d" , "--direction"     , **argv, type=flist, help="direction of the displacement")
    parser.add_argument("-o" , "--output"        , **argv, type=str  , help="output file (default: %(default)s)", default="path.extxyz")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #-------------------#
    print("\tReading first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,index=0)[0]
    print("done")

    #-------------------#
    print("\tSelecting atoms ... ", end="")
    indices = AtomSelector.select(args.atoms,atoms)
    print("done")
    print("\tselected atoms: ",indices)

    #-------------------#
    print("\tEvaluating replicas position ... ", end="")
    pos = atoms.get_positions()[indices,:]
    frac = cart2frac(atoms.get_cell(),pos)
    frac += args.direction
    disp_pos = frac2cart(atoms.get_cell(),frac)
    replica = atoms.copy()
    tmp = replica.get_positions()
    tmp[indices,:] = disp_pos
    replica.set_positions(tmp)
    print("done")

    #-------------------#
    print("\tCreating the path ... ", end="")
    N = args.number+2
    path = [ atoms.copy() for _ in range(N) ]
    posA = atoms.get_positions()
    posB = replica.get_positions()
    for n in range(N):
        t = float(n)/(N-1)
        pos = (1-t)*posA + t*posB
        path[n].set_positions(pos)
    print("done")

    #-------------------#
    print("\n\tWriting displaced structures to file '{:s}' ... ".format(args.output), end="")
    path = AtomicStructures(path)
    path.to_file(file=args.output)
    print("done")
    

if __name__ == "__main__":
    main()

