#!/usr/bin/env python
from ase.io import read
from eslib.formatting import esfmt, warning
from ase import Atoms
import numpy as np
from ase.io import read, write
from eslib.tools import cart2frac, frac2cart
from eslib.input import slist
from typing import List
from classes.atomic_structures import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Divide a system into (water) molecules."
documentation = "This script is mainly targeted for bulk water systems"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-rc", "--cutoff"       , **argv, required=False, type=float, help="cutoff/bond length(default: %(default)s)" , default=1.2)
    parser.add_argument("-n" , "--n_bonds"      , **argv, required=False, type=int  , help="number of bonds (default: %(default)s)", default=2)
    parser.add_argument("-s" , "--species"      , **argv, required=False, type=slist, help="atomic species of the bonds to be fixed (default: %(default)s)", default=['O','H'])
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str  , help="molecule name (default: %(default)s)", default="molecule")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file with the oxidation numbers (default: %(default)s)", default="wrapped.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if len(args.species) != 2:
        raise ValueError("-s,--species has to be of length 2.")
    print("\tFixing bonds between {:s} and {:s}".format(args.species[0],args.species[1]))
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tNumber atomic structures: ",len(trajectory))

    #------------------#
    # Select oxygen and hydrogen atoms
    oxygens   = [ n for n,a in enumerate(trajectory[0]) if a.symbol == args.species[0]]
    hydrogens = [ n for n,a in enumerate(trajectory[0]) if a.symbol == args.species[1]]
    print()
    print("\tNumber of {:s} atoms: {:d}".format(args.species[0],len(oxygens)))
    print("\tNumber of {:s} atoms: {:d}".format(args.species[1],len(hydrogens)))


    #------------------#
    print("\tDividing into molecules:")
    zeros = np.zeros(3)
    N = len(trajectory)
    for n,atoms in enumerate(trajectory):
        print("\t - atomic structure {:d}/{:d}".format(n+1,N),end="\r")
        oxygens   = [ n for n,a in enumerate(atoms) if a.symbol == args.species[0]]
        hydrogens = [ n for n,a in enumerate(atoms) if a.symbol == args.species[1]]

        Natoms = atoms.get_global_number_of_atoms()
        molecule = np.full(Natoms,np.nan)
        for mm,o_index in enumerate(oxygens):
            # Find neighbors of the current oxygen atom within the cutoff distance

            if not np.isnan(molecule[o_index]):
                raise ValueError("coding error")
            molecule[o_index] = mm
            
            distances = atoms.get_distances(o_index,hydrogens,mic=True,vector=False)
            indices = list(np.argsort(distances)[:args.n_bonds])
            indices = np.asarray(hydrogens)[indices]

            if args.cutoff is not None:
                distances.sort()
                count = (distances < args.cutoff ).sum()
                if count != args.n_bonds:
                    pass

            for i in indices:
                if not np.isnan(molecule[i]):
                    raise ValueError("coding error")
                molecule[i] = mm

        if atoms.arrays is None:
            atoms.arrays = dict()
        if np.any(np.isnan(molecule)):
            raise ValueError("coding error")
        atoms.arrays[args.molecule] = molecule.astype(int)

    print("\n\tWriting (un)wrapped atomic structure to file '{:s}' ... ".format(args.output), end="")
    trajectory.to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()

# {
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/convert/fix-water-bonds.py",
#             "cwd" : "/home/stoccoel/google-personal/works/water/MACE/",
#             "console": "integratedTerminal",
#             "justMyCode": false,
#             "args": [
#                 "-i", "MACE_test.extxyz", 
#                 // "-m", "model/dipole_32.model", 
#                 // "-p", "your_port_number", 
#                 // "-a", "your_address", 
#                 // "-d", "cpu"
#             ],
#             "env": {
#                 "PYTHONPATH": "/home/stoccoel/google-personal/codes/mace/mace/"
#             }
#         }
#     ]
# }


