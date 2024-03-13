#!/usr/bin/env python
from ase.io import read
from eslib.formatting import esfmt
from ase import Atoms, Atom
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, neighbor_list

#---------------------------------------#
# Description of the script's purpose
description = "MACE socket driver."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: 'None')" , default=None)
    parser.add_argument("-rc", "--cutoff"      , **argv, required=False, type=float, help="cutoff/bond lenght [bohr] (default: 3)" , default=3)
    # parser.add_argument("-m", "--model"     , **argv, required=True , type=str, help="file with the MACE model")
    # parser.add_argument("-t", "--model_type", **argv, required=False, type=str, help="MACE model data type (default: None)", default=None)
    # parser.add_argument("-p", "--port"      , **argv, required=True , type=str, help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    # parser.add_argument("-a", "--address"   , **argv, required=True , type=str, help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    # parser.add_argument("-u", "--unix"      , **argv, required=True , type=str, help="Use a UNIX domain socket.")
    # parser.add_argument("-d", "--device"    , **argv, required=True , type=str, help="device (default: 'cpu')", choices=['cpu','gpu','cuda'], default='cpu')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = read(args.input,format=args.input_format,index=0)
    print("done")

    # Define a cutoff distance
    cutoff = args.cutoff  # Angstrom

    # Create a neighbor list
    nl = NeighborList([cutoff / 2.0] * len(atoms), self_interaction=False, bothways=True)

    # Build the neighbor list
    nl.update(atoms)

    # Select oxygen and hydrogen atoms
    oxygens = [ a for a in atoms if a.symbol == 'O']

    # Find hydrogen atoms near oxygen atoms
    oxygen_hydrogen_pairs = []

    for o_atom in oxygens:
        # Find neighbors of the current oxygen atom within the cutoff distance
        neighbor_list
        indices, offsets = nl.get_neighbors(o_atom.index)
        
        for idx, offset in zip(indices, offsets):
            if atoms[idx].symbol == 'H':
                distance = np.linalg.norm(np.dot(offset, atoms.get_cell()))  # Calculate distance with PBC
                if distance < cutoff:
                    oxygen_hydrogen_pairs.append((o_atom.index, idx,distance))

    # Print the pairs of oxygen and nearby hydrogen atoms
    for pair in oxygen_hydrogen_pairs:
        print("Oxygen atom:", pair[0], "Hydrogen atom:", pair[1], "Distance:",pair[2])

    pass

#---------------------------------------#
if __name__ == "__main__":
    main()



