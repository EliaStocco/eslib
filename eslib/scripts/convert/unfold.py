#!/usr/bin/env python
import argparse

import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import cart2frac, frac2cart

#---------------------------------------#
# Description of the script's purpose
description = "Unfold an MD trajectory of a periodic system."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv,type=str, help="input file")
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m"  , "--method"       ,   **argv,type=str, help="method (default: %(default)s)", default='eslib', choices=['ase','eslib'])    
    parser.add_argument("-o"  , "--output"       ,   **argv,type=str, help="output file")
    parser.add_argument("-of" , "--output_format",   **argv,type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from input file '{:s}' ... ".format(args.input), end="")
    structures:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    
    #------------------#
    N = len(structures)
    shape = structures[0].positions.shape
    print("\t{:s}: {:d}".format("Number of atomic structures",N))
    print("\t{:s}: ({:d},{:d})".format("Positions shape of each structure",shape[0],shape[1]))

    if args.method == 'ase':
        #------------------#
        print("\n\tComputing fractional/scaled coordinates ... ", end="")
        positions = np.full((N,*shape),np.nan)
        for n in range(len(structures)):
            positions[n,:,:] = structures[n].get_scaled_positions()
        print("done")

        #------------------#
        print("\tUnfolding fractional/scaled coordinates ... ", end="")
        positions = positions.reshape((N,-1))
        unfolded_positions = np.unwrap(positions,axis=0,period=1)
        print("done")

        #------------------#
        print("\tSetting unfolded fractional/scaled coordinates ... ", end="")
        unfolded_positions = unfolded_positions.reshape((N,*shape))
        for n in range(len(structures)):
            structures[n].set_scaled_positions(unfolded_positions[n,:,:])
        print("done")

        #------------------#
        bool2str = lambda value: "true" if value else "false"
        positions = positions.reshape((N,-1))
        a = positions.reshape((N,-1))
        b = unfolded_positions.reshape((N,-1))
        diff = np.diff(a-b,axis=0)
        modified = np.square(diff).sum(axis=1) > 0.1
        index = np.where(modified == True)[0]
        unfolded = len(index) > 0 
        tf = bool2str(unfolded)
        print("\n\tNumber of structures that have been unfolded: {:d}".format(len(index)))
        # print("\n\tAt least one coordinate has been unfolded: {:s}".format(tf))
        # if unfolded:
        #     modified = (np.sqrt(np.square(a-b).sum(axis=1)) > 0.1) # whether they have been modified
        #     index = np.where(modified == True)[0]

    elif args.method == 'eslib':

        print("\n\tComputing fractional/scaled coordinates ... ", end="")
        frac_positions = np.full((N,*shape),np.nan)
        for n,atoms in enumerate(structures):
            frac_positions[n,:,:] = cart2frac(atoms.get_cell(),atoms.get_positions())
        print("done")

        #------------------#
        # print("\tUnfolding fractional/scaled coordinates ... ", end="")
        # frac_positions = frac_positions.reshape((N,-1))
        # frac_positions = np.mod(frac_positions,1)
        # frac_positions = np.unwrap(frac_positions,axis=0,period=1)
        # print("done")

        # Nfolded = np.any( (frac_positions != unfolded_frac_positions ).reshape((len(atoms),-1)) , axis=1).sum()
        # print("\n\tNumber of structures that have been unfolded: {:d}".format(Nfolded))

        #------------------#
        print("\tSetting cartesian coordinates ... ", end="")
        for n,atoms in enumerate(structures):
            frac = np.asarray(frac_positions[n]).reshape((-1,3))
            positions = frac2cart(atoms.get_cell(),frac)
            atoms.set_positions(positions,apply_constraint=False)
        print("done")


    #------------------#
    print("\n\tWriting unfolded structures to output file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()