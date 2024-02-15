#!/usr/bin/env python
import numpy as np
import os
from eslib.nn.functions import get_model
from ase.io import read
from ase.geometry import wrap_positions
from eslib.formatting import esfmt
from eslib.input import str2bool
from ase import Atoms

#---------------------------------------#
# Description of the script's purpose
description = "Check the E(3)-equivariance of a neural network."

def prepare_args(description):
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--instructions", type=str, **argv, help="model input file (default: 'instructions.json')", default="instructions.json")
    parser.add_argument("-p" , "--parameters"  , type=str, **argv, help="torch parameters file (default: 'parameters.pth')", default=None)
    parser.add_argument("-s" , "--structure"   , type=str, **argv, help="file with an atomic structure [a.u.]")
    parser.add_argument("-n" , "--number"      , type=int, **argv, help="number of tests to perform", default=100)
    parser.add_argument("-f" , "--fold"        , type=str2bool, **argv, help="whether the atomic structures have to be folded into the primitive unit cell (default: false)", default=False)
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.structure), end="")
    atoms = read(args.structure)
    print("done")

    #------------------#
    print("\tLoading model ... ",end="")
    file_in = os.path.normpath("{:s}".format(args.instructions))
    file_pa = os.path.normpath("{:s}".format(args.parameters)) if args.parameters is not None else None
    model = get_model(file_in,file_pa)
    print("done")

    #------------------#

    pos = atoms.positions.reshape((-1,3))
    # in i-PI format
    pbc = np.all(atoms.get_pbc())
    cell = atoms.get_cell() if pbc else None
    if pbc and args.fold:
        pos = wrap_positions(positions=pos,cell=cell)

    #------------------#
    def permute_atoms(structure:Atoms):
        # Get atomic positions and symbols
        positions = structure.get_positions()
        symbols = structure.get_chemical_symbols()
        
        # Combine positions and symbols
        Natoms = structure.get_global_number_of_atoms()
        # combined = list(zip(positions, symbols))
        indices = np.arange(Natoms)
        
        # Shuffle combined list
        np.random.shuffle(indices)
        
        # Unpack shuffled positions and symbols
        # shuffled_positions, shuffled_symbols = zip(*combined)
        shuffled_positions = positions[indices]
        shuffled_symbols   = [ symbols[i] for i in  indices ]
        
        # Create permuted structure
        permuted_structure = Atoms( positions=shuffled_positions, \
                                    symbols=shuffled_symbols, \
                                    cell=structure.get_cell(), \
                                    pbc=structure.get_pbc())
        
        return permuted_structure

    #------------------#
    print("\tComparing 'outputs from permuted inputs' with 'outputs' ... ",end="")
    y,_ = model.get(pos=pos.reshape((-1,3)),cell=cell)
    y = y.detach().numpy()
    norm = np.zeros(args.number)
    for n in range(args.number):
        patoms = permute_atoms(atoms)
        Tx2y, _ = model.get_from_structure(patoms) 
        Tx2y = Tx2y.detach().numpy()
        norm[n] = np.linalg.norm(Tx2y - y)
    print("done")
    
    print("\tSummary of the norm between 'outputs from translated inputs' and 'outputs'")
    print("\t{:>20s}: {:.4e}".format("min norm",norm.min()))
    print("\t{:>20s}: {:.4e}".format("max norm",norm.max()))
    print("\t{:>20s}: {:.4e}".format("mean norm",norm.mean()))

    return norm

#---------------------------------------#
if __name__ == "__main__":
    main()
