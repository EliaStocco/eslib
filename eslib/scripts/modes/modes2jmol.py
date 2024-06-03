#!/usr/bin/env python
from eslib.tools import convert
import argparse
from ase.io import read
import numpy as np
from eslib.formatting import esfmt

#---------------------------------------#
""" 
13.12.2023 Elia Stocco
    Some changes to the previous script:
    - using 'argparse'
    - verbose output to screen
    - defined a function callable from other scripts

04.05.2020 Karen Fidanyan
    This script takes the relaxed geometry in XYZ format
    and the .mode, .eigval files produced by i-PI,
    and builds a .xyz_jmol file to visualize vibrations.

"""
#---------------------------------------#
# Description of the script's purpose
description = "Create a JMOL file with the normal modes from the '*.mode' output file of a i-PI vibrational analysis."

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i",  "--input",        type=str, **argv, help="atomic structure file [angstrom,xyz]")
    parser.add_argument("-m",  "--modes",        type=str, **argv, help="file with vibrational modes displacements [a.u.] (default: %(default)s)", default="i-pi.phonons.mode")
    parser.add_argument("-w",  "--eigenvalues",  type=str, **argv, help="file with vibrational modes eigenvalues [a.u.] (default: %(default)s)", default=None)
    parser.add_argument("-o",  "--output",       type=str, **argv, help="JMOL output file (default: %(default)s)", default="vibmodes.jmol")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # read input file
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    #---------------------------------------#
    # read vibrational modes
    print("\tReading vibrational modes displacements from file '{:s}' ... ".format(args.modes), end="")
    modes = np.loadtxt(args.modes)
    print("done")

    if atoms.positions.flatten().shape[0] != modes.shape[0]:
        raise ValueError("positions and modes shapes do not match.")

    #---------------------------------------#
    # read eigenvalues
    if args.eigenvalues is not None:
        print("\tReading vibrational modes eigenvalues from file '{:s}' ... ".format(args.eigenvalues), end="")
        eigvals = np.loadtxt(args.eigenvalues)
        print("done")
    else:
        print("\tNo vibrational modes eigenvalues provided: setting them to zero ... ", end="")
        eigvals = np.zeros(modes.shape[0])
        print("done")

    if eigvals.shape[0] != modes.shape[0]:
        raise ValueError("eigvals and modes shapes do not match.")

    #---------------------------------------#
    # frequencies
    print("\tComputing frequencies ... ", end="")
    freqs = np.sqrt(eigvals) 
    freqs = convert(freqs,"frequency","atomic_unit","thz")
    print("done")

    #---------------------------------------#
    # write JMOL file
    print("\tWriting vibrational modes to file '{:s}' ... ".format(args.output), end="")
    np.set_printoptions(formatter={'float': '{: .8f}'.format})
    with open(args.output, 'w') as fdout:
        for b, vec in enumerate(modes.T):
            disp = vec.reshape(-1, 3)
            fdout.write("%i\n# %f THz, branch # %i\n"
                        % (len(atoms), freqs[b], b))
            for i, atom in enumerate(atoms.positions):
                fdout.write("%s  " % atoms[i].symbol
                            + ' '.join(map("{:10.8g}".format, atom)) + "  "
                            + ' '.join(map("{:12.8g}".format, disp[i])) + "\n")
    print("done")

if __name__ == "__main__":
    main()
