#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.plot import plot_bisector
from eslib.classes.models.dipole.baseclass import DipoleModel, DipoleLinearModel
from eslib.classes.models.dipole import DipolePartialCharges
from eslib.classes.atomic_structures import info
from eslib.classes.atomic_structures import AtomicStructures
from eslib.tools import cart2lattice, cart2frac, frac2cart
from eslib.output import output_folder
from ase.io import write
from eslib.formatting import esfmt, everythingok, warning, error
import json
from eslib.show import show_dict
from eslib.tools import is_integer
from eslib.geometry import fold

#---------------------------------------#
# Description of the script's purpose
description = "Fix the dipole jumps of a folded trajectory using the oxidation numbers."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv, type=str, help="extxyz file with the unfolded atomic configurations [a.u]")
    parser.add_argument("-id", "--in_dipole"  , **argv, type=str, help="name of the input dipoles(default: %(default)s)", default='dipole')
    parser.add_argument("-od", "--out_dipole"  , **argv, type=str, help="name of the output dipoles(default: %(default)s)", default='dipole')
    parser.add_argument("-c", "--charges", **argv, type=str, required=True , help="JSON file with the oxidation number/integer charges")
    parser.add_argument("-o", "--output"   , **argv, type=str, help="output file with the fixed trajectory (default: %(default)s)", default="trajectory.fixed.extxyz")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the oxidation numbers from file '{:s}' ... ".format(args.charges), end="")
    with open(args.charges, 'r') as json_file:
        charges:dict = json.load(json_file)
    print("done")

    #------------------#
    print("\n\tOxidation numbers: ")
    show_dict(charges,"\t\t",2)

    for k,c in charges.items():
        if not is_integer(c):
            raise ValueError("\t{:s}: '{:s}' charge is not an integer".format(warning,k))

    #------------------#
    print("\n\tCreating dipole model based on the partial charges ... ",end="")
    model = DipolePartialCharges(charges)
    print("done")

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")
    print("\tn. of atomic structures: ",len(trajectory))

    #------------------#
    print("\n\tFolding the atomic structures ... ",end="")
    trajectory, shift = fold(trajectory)
    print("done")

    #------------------#
    print("\n\tComputing the dipole shifts using the oxidation numbers ... ",end="")
    for n,atoms in enumerate(trajectory):
        oxn = model.get_all_charges(atoms)
        quanta_shift = oxn @ shift[n]
        dipole_shift = frac2cart(atoms.get_cell(),quanta_shift)
        atoms.info[args.out_dipole] = atoms.info[args.in_dipole] + dipole_shift
    print("done")

    #------------------#
    print("\n\tWriting output to file '{:s}' ... ".format(args.output), end="")
    trajectory.to_file(file=args.output)
    print("done")
    

#---------------------------------------#
if __name__ == "__main__":
    main()