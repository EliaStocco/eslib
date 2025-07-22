#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.geometry import get_distances, conditional_find_mic
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Filter an extxyz."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-r" , "--reference"         , **argv, required=True , type=str  , help="file with the reference structure")
    parser.add_argument("-rf", "--reference_format"  , **argv, required=False, type=str  , help="reference file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"     , **argv, required=False, type=str, help="keyword (default: %(default)s)", default="distance")
    parser.add_argument("-o" , "--output"       , required=True,**argv,type=str, help="txt output file")
    parser.add_argument("-of", "--output_format", required=False,**argv,type=str, help="txt output format for np.savetxt (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from input file '{:s}' ... ".format(args.input), end="")
    structure = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tNumber of structures: ", len(structure))
    Natoms =structure.num_atoms()
    print("\tNumber of atoms: ",Natoms)
    
    #------------------#
    print("\tReading the reference structure from input file '{:s}' ... ".format(args.reference), end="")
    reference:Atoms = AtomicStructures.from_file(file=args.reference,format=args.reference_format,index=0)[0]
    print("done")
    
    assert reference.get_global_number_of_atoms() == Natoms, \
        "Reference structure must have the same number of atoms as the input structure"

    #------------------#
    print("\tComputing distances ... ", end="")
    com = reference.get_center_of_mass()
    cell = reference.get_cell()
    distances = np.zeros((len(structure), Natoms), dtype=np.float64)
    for n,_atoms in enumerate(structure):
        atoms = _atoms.copy()
        atoms.set_center_of_mass(com)
        delta = atoms.get_positions() - reference.get_positions()
        distances[n]  = conditional_find_mic(delta, cell=cell, pbc=reference.get_pbc())[1]
    print("done")
    print("\tdistances.shape: ", distances.shape)

    #------------------#
    print("\tStoring distances  ... ", end="")
    structure.set(args.keyword, distances,"arrays")
    print("done")
    
    #------------------#
    print("\tStoring mean distances ... ", end="")
    mean_distances = np.mean(distances, axis=1)
    structure.set(f"{args.keyword}_mean", mean_distances,"info")
    print("done")
    
    #------------------#
    print("\tStoring max distances ... ", end="")
    max_distances = np.max(distances, axis=1)
    structure.set(f"{args.keyword}_max", max_distances,"info")
    print("done")
    
    #------------------#
    print("\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    structure.to_file(file=args.output, format=args.output_format)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()

