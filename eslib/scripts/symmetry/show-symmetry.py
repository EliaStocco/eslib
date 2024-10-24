#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import numpy as np
import spglib
from ase import Atoms
from ase.spacegroup import get_spacegroup
from ase.spacegroup.spacegroup import Spacegroup
from joblib import PrintTime
from phonopy.cui.show_symmetry import _get_symmetry_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (Primitive, get_primitive,
                                     guess_primitive_matrix)
from phonopy.structure.symmetry import Symmetry

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist

#---------------------------------------#
description = "Find the symmetry of an atomic structure."
    
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"       , type=str  , **argv, required=True , help="atomic structure input file")
    parser.add_argument("-if", "--input_format", type=str  , **argv, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-t" , "--threshold"   , type=float, **argv, required=False, help="list of thresholds (default: %(default)s)" , default=1e-3)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    dataset = spglib.get_symmetry_dataset(cell,args.threshold)
    print(dataset.__dict__.keys())

    #------------------#
    print()
    line = "|{:^15s}|{:^15s}|{:^15s}|{:^15s}|".format("Threshold","Spacegroup","Spacegroup","n. of sym.")
    N = len(line)-2
    print("\t|"+"-"*N+"|")
    print("\t"+line)
    line = "|{:^15s}|{:^15s}|{:^15s}|{:^15s}|".format("","symbol","number","operations")
    print("\t"+line)
    print("\t|"+"-"*N+"|")
    spacegroup:Spacegroup = get_spacegroup(atoms,symprec=args.threshold)
    line = "|{:>12.2e}   |{:^15s}|{:^15d}|{:^15d}|".format(args.threshold,spacegroup.symbol,spacegroup.no,spacegroup.nsymop)
    print("\t"+line)
        # print("\tThreshold: {:>.2e}  Spacegroup: {:>6s}".format(symprec,spacegroup.symbol,spacegroup.no,spacegroup.nsymop))
    print("\t|"+"-"*N+"|")
    
    frac = atoms.get_scaled_positions()
    
    PhAtoms  = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell
    )
    
    symmetry = Symmetry(PhAtoms)
    
    # cs = check_symmetry(phonon=PhAtoms)
    
    text:str = _get_symmetry_yaml(cell=PhAtoms,symmetry=symmetry)
    text = "\n\t" + text.replace("\n","\n\t")
    print(text)
    
    # def _expand_borns(borns, primitive: PhonopyAtoms, prim_symmetry: Symmetry):
    #     # Expand Born effective charges to all atoms in the primitive cell
    #     rotations = prim_symmetry.symmetry_operations["rotations"]
    #     map_operations = prim_symmetry.get_map_operations()
    #     map_atoms = prim_symmetry.get_map_atoms()

    #     for i in range(len(primitive)):
    #         # R_cart = L R L^-1
    #         rot_cartesian = similarity_transformation(
    #             primitive.cell.T, rotations[map_operations[i]]
    #         )
    #         # R_cart^T B R_cart^-T (inverse rotation is required to transform)
    #         borns[i] = similarity_transformation(rot_cartesian.T, borns[map_atoms[i]])
    
    symmetry.get_site_symmetry(0)
    
    ia = symmetry.get_independent_atoms()
    
    
    trans_mat = guess_primitive_matrix(PhAtoms, symprec=args.threshold)
    primitive:Primitive = get_primitive(PhAtoms, trans_mat, symprec=args.threshold)
    
    print("\nSymmetry operations:")
    for i, (rot, trans) in enumerate(spacegroup.get_symop()):
        print()
        print("\t#{}".format(i + 1))
        print("\tRotation matrix: ")
        print("                 ", rot[0,:])
        print("                 ", rot[1,:])
        print("                 ", rot[2,:])
        print("\tTranslation vector: ")
        print("                 ", trans)
        rot@frac+trans
    
    
    out = spacegroup.equivalent_sites(frac)
    out = spacegroup.symmetry_normalised_sites(frac)
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()