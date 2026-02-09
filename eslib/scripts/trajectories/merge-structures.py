#!/usr/bin/env python
import numpy as np
from ase.io import write
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Merge two extxyz structures."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-a" , "--structure_A"        , **argv, required=True , type=str  , help="input file A")
    parser.add_argument("-b" , "--structure_B"        , **argv, required=True , type=str  , help="input file B")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structure from file '{:s}' ... ".format(args.structure_A), end="")
    snapshot_A:Atoms = AtomicStructures.from_file(file=args.structure_A,format="extxyz",index=0)[0]
    print("done")
    
    print("\tReading atomic structure from file '{:s}' ... ".format(args.structure_B), end="")
    snapshot_B:Atoms = AtomicStructures.from_file(file=args.structure_B,format="extxyz",index=0)[0]
    print("done")

    #---------------------------------------#
    Na = snapshot_A.get_global_number_of_atoms()
    Nb = snapshot_B.get_global_number_of_atoms()
    assert Na == Nb, f"Different number of atoms: {Na} != {Nb}."
    
    posA = snapshot_A.get_positions()
    posB = snapshot_B.get_positions()
    
    assert np.allclose(posA,posB), "Different positions"
    
    cellA = snapshot_A.get_cell().cellpar()
    cellB = snapshot_B.get_cell().cellpar()
    
    assert np.allclose(cellA,cellB), "Different cells"
    
    print("\tGeometries are identical.\n")
    
    #---------------------------------------#
    infoA = list(snapshot_A.info.keys())
    infoB = list(snapshot_B.info.keys())
    arrA = list(snapshot_A.arrays.keys())
    arrB = list(snapshot_B.arrays.keys())
    for x in [arrA,arrB]:
        for a in ['positions','numbers']:
            x.remove(a)
    for x in [infoA,infoB,arrA,arrB]:
        x.sort()
    
    print("\tStructure A -- info:   ",infoA)
    print("\tStructure B -- info:   ",infoB)
    print("\tStructure A -- arrays: ",arrA)
    print("\tStructure B -- arrays: ",arrB)
    
    #---------------------------------------#
    print("\n\tAdding to structure A the poperties of B ... ",end="")
    for k in infoB:
        snapshot_A.info[k] = snapshot_B.info[k]
    for k in arrB:
        snapshot_A.arrays[k] = snapshot_B.arrays[k]
    print("done")
    
    info = list(snapshot_A.info.keys())
    arr = list(snapshot_A.arrays.keys())
    arr.remove('positions')
    arr.remove('numbers')
    print("\tStructure -- info: ",info)
    print("\tStructure -- arrays: ",arr)
    
    #---------------------------------------#
    print("\n\tWriting merged structure to file '{:s}' ... ".format(args.output), end="")
    write(args.output, snapshot_A)
    print("done")
    
    return
    

if __name__ == "__main__":
    main()

