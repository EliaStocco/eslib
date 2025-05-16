#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.input import str2bool
from eslib.tools import cart2frac, frac2cart

#---------------------------------------#
# Description of the script's purpose
description = "Wrap hydrogen atoms such that they will be close to an oxygen atoms."
documentation = \
"This script is targeted for bulk water systems.\n\
The options -c/--check checks that the folding procedure does not modify the interatomic distances.\n\
Pay attention that this flag can slow down a lot the script."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-rc", "--cutoff"       , **argv, required=False, type=float, help="cutoff/bond length(default: %(default)s)" , default=1.2)
    parser.add_argument("-n" , "--n_bonds"      , **argv, required=False, type=int  , help="number of bonds (default: %(default)s)", default=2)
    parser.add_argument("-c" , "--check"        , **argv, required=False, type=str2bool, help="check that interatomic distances are the same (default: %(default)s)", default=False)
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file with the oxidation numbers (default: %(default)s)", default="wrapped.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tNumber atomic structures: ",len(trajectory))
    
    assert trajectory.is_trajectory(), "The input file should contain a trajectory."
    
    species = trajectory.get_chemical_symbols(unique=True)
    print("\tUnique atomic species: ",species)
    species = set(species)
    assert species == set(["O","H"]), "The input file should contain only O and H atoms."

    #------------------#
    print("\tWrapping the first snapshot into the unit cell ... ", end=" ")
    frac = trajectory[0].get_scaled_positions()
    frac = frac % 1
    trajectory[0].set_scaled_positions(frac)
    print("done")
    
    #------------------#
    print("\tFixing bonds of the first snapshot ... ", end=" ")
        
    oxygens   = [ n for n,a in enumerate(trajectory[0]) if a.symbol == "O"]
    hydrogens = [ n for n,a in enumerate(trajectory[0]) if a.symbol == "H"]

    if args.check:
        all_distances = trajectory[0].get_all_distances(mic=True,vector=False)

    wrapped = []
    for o_index in oxygens:

        if args.check:
            distances = all_distances[hydrogens,o_index]
        else:
            distances = trajectory[0].get_distances(o_index,hydrogens,mic=True,vector=False)
        indices = list(np.argsort(distances)[:args.n_bonds])

        if args.cutoff is not None:
            distances.sort()
            count = (distances < args.cutoff ).sum()
            if count != args.n_bonds:
                pass

        for n in np.asarray(hydrogens)[indices]:
            delta = trajectory[0].positions[n] - trajectory[0].positions[o_index] 
            delta:np.ndarray = cart2frac(trajectory[0].get_cell(),delta)
            delta = delta.round(0).astype(int)
            if not np.allclose(delta,np.zeros(3)):
                delta = frac2cart(trajectory[0].get_cell(),delta)
                trajectory[0].positions[n,:] -= delta
                wrapped.append(n)

        
    if args.check:
        new_distances = trajectory[0].get_all_distances(mic=True,vector=False)
        assert np.allclose(all_distances,new_distances)

    Nwrapping = len(wrapped)
    Nwrapped  = len(np.unique(wrapped))

    print("done")

    if Nwrapping != Nwrapped:
        print("\t{:s}: the previous two numbers are expected to be the same. Carefully check your input and output files.".format(warning))

    #------------------#
    print("\tUnwrapping the remaining snapshots ... ", end=" ")
    trajectory.unwrap(inplace=True)
    print("done")
    
    #------------------#
    print("\n\tWriting unwrapped atomic structure to file '{:s}' ... ".format(args.output), end="")
    trajectory.to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
