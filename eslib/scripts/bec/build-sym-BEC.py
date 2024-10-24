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
from phonopy import Phonopy
from phonopy.cui.show_symmetry import _get_symmetry_yaml
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS
from phonopy.harmonic.force_constants import (
    _get_force_constants_disps, distribute_force_constants,
    distribute_force_constants_by_translations)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (Primitive, get_primitive,
                                     guess_primitive_matrix)
from phonopy.structure.symmetry import Symmetry

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist


def get_bec(
    supercell: PhonopyAtoms,
    symmetry: Symmetry,
    dataset,
    atom_list=None,
    primitive=None,
    decimals=None,
):
    """Force constants are computed.

    Force constants, Phi, are calculated from sets for forces, F, and
    atomic displacement, d:
      Phi = -F / d
    This is solved by matrix pseudo-inversion.
    Crystal symmetry is included when creating F and d matrices.

    Returns
    -------
    ndarray
        Force constants[ i, j, a, b ]
        i: Atom index of finitely displaced atom.
        j: Atom index at which force on the atom is measured.
        a, b: Cartesian direction indices = (0, 1, 2) for i and j, respectively
        dtype=double
        shape=(len(atom_list),n_satom,3,3),

    """
    if atom_list is None:
        fc_dim0 = len(supercell)
    else:
        fc_dim0 = len(atom_list)

    force_constants = np.zeros(
        (fc_dim0, len(supercell), 3, 3), dtype="double", order="C"
    )

    # Fill force_constants[ displaced_atoms, all_atoms_in_supercell ]
    atom_list_done = _get_force_constants_disps(
        force_constants, supercell, dataset, symmetry, atom_list=atom_list
    )
    rotations = symmetry.symmetry_operations["rotations"]
    lattice = np.array(supercell.cell.T, dtype="double", order="C")
    permutations = symmetry.atomic_permutations

    if atom_list is None and primitive is not None:
        # Distribute to atoms in primitive cell, then distribute to all.
        distribute_force_constants(
            force_constants,
            atom_list_done,
            lattice,
            rotations,
            permutations,
            atom_list=primitive.p2s_map,
            fc_indices_of_atom_list=primitive.p2s_map,
        )
        distribute_force_constants_by_translations(
            force_constants, primitive, supercell
        )
    else:
        distribute_force_constants(
            force_constants,
            atom_list_done,
            lattice,
            rotations,
            permutations,
            atom_list=atom_list,
        )

    if decimals:
        force_constants = force_constants.round(decimals=decimals)

    return force_constants



#---------------------------------------#
description = "Find the symmetry of an atomic structure."
    
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-s" , "--supercell"       , type=str  , **argv, required=True , help="supercell atomic structure input file")
    parser.add_argument("-sf", "--supercell_format", type=str  , **argv, required=False, help="primitive structure file format (default: %(default)s)" , default=None)
    parser.add_argument("-d" , "--displaced"       , type=str  , **argv, required=True , help="displaced atomic structure input file")
    parser.add_argument("-df", "--displaced_format", type=str  , **argv, required=False, help="displaced structures file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--name"            , type=str  , **argv, required=True , help="dipole name (default: %(default)s)" , default='dipole')
    parser.add_argument("-t" , "--threshold"       , type=float, **argv, required=False, help="list of thresholds (default: %(default)s)" , default=1e-3)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading the primitive atomic structure from file '{:s}' ... ".format(args.supercell), end="")
    supercell:Atoms = AtomicStructures.from_file(file=args.supercell,format=args.supercell_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",supercell.get_global_number_of_atoms())
    
    #------------------#
    print("\n\tReading the displaced atomic structures from file '{:s}' ... ".format(args.displaced), end="")
    displaced:AtomicStructures = AtomicStructures.from_file(file=args.displaced,format=args.displaced_format)
    print("done")
    print("\tn. of atoms: ",displaced[0].get_global_number_of_atoms())
    print("\tn. of structures: ",len(displaced))
    
    # assert np.all( [a.get_global_number_of_atoms() == supercell.get_global_number_of_atoms() for a in displaced ] ), "atomic structures have different numbers of atoms"
    
    #------------------#
    print("\n\tExtracting dipole ... ", end="")
    dipole = displaced.get(args.name)
    print("done")
    
    Z = (dipole[1] - dipole[0])/(2*0.01)
    
    #------------------#
    print("\n\tBuilding PhonopyAtoms ... ", end="")
    frac = supercell.get_scaled_positions()
    PhAtoms  = PhonopyAtoms(
        symbols=supercell.get_chemical_symbols(),
        scaled_positions=supercell.get_scaled_positions(),
        cell=supercell.cell
    )
    print("done")
    
    #------------------#
    print("\tBuilding Symmetry ... ", end="")
    symmetry = Symmetry(PhAtoms,symprec=args.threshold)
    print("done")
    
    phonon = Phonopy(
        unitcell = PhAtoms,
        supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]],
        # primitive_matrix="P",
        symprec=args.threshold,
        )

    symmetry = phonon.get_symmetry()
    print("Space group: %s" % symmetry.get_international_table())
    
    force_sets = parse_FORCE_SETS()
    phonon.dataset = force_sets
    
    get_bec(phonon._supercell,
            phonon._symmetry,
            phonon._displacement_dataset,
            atom_list=None,
            primitive=phonon._primitive,
            decimals=None
            )

    force_sets = parse_FORCE_SETS()
    phonon.dataset = force_sets
    phonon.produce_force_constants()


    disps, forces = get_displacements_and_forces(self._displacement_dataset)
    
    indep_atoms = symmetry.get_independent_atoms()
    
    print("\n\tIndependent atoms: ",indep_atoms)
    
    print("\n\tN. of site symmetries:")
    for n,i in enumerate(indep_atoms):
        site_sym = symmetry.get_site_symmetry(i)
        print("\t - indep. atom {:d}: {:d}".format(i,len(site_sym)))
        
    

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
    
    frac = supercell.get_scaled_positions()
    
    PhAtoms  = PhonopyAtoms(
        symbols=supercell.get_chemical_symbols(),
        scaled_positions=supercell.get_scaled_positions(),
        cell=supercell.cell
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