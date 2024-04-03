#!/usr/bin/env python
from ase.io import read, write
import argparse
import numpy as np
from eslib.formatting import esfmt, error, warning
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Fold the atomic structures into the primitive cell."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv,type=str, help="input file")
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-o"  , "--output"       ,   **argv,type=str, help="output file")
    parser.add_argument("-of" , "--output_format",   **argv,type=str, help="output file format (default: 'None')", default=None)
    # options = parser.parse_args()
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from input file '{:s}' ... ".format(args.input), end="")
    # atoms = read(args.input,index=":",format=args.input_format)
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    N = len(atoms)
    shape = atoms[0].positions.shape
    print("\t{:s}: {:d}".format("Number of atomic structures",N))
    print("\t{:s}: ({:d},{:d})".format("Positions shape of each structure",shape[0],shape[1]))

    #------------------#
    print("\n\tComputing fractional/scaled coordinates into the primitive cell ... ", end="")
    
    positions = np.full((N,*shape),np.nan)
    for n in range(len(atoms)):
        positions[n,:,:] = atoms[n].get_scaled_positions(wrap=False)
    print("done")

    #------------------#
    print("\tFolding fractional/scaled coordinates ... ", end="")
    positions = positions.reshape((N,-1))
    folded_positions = np.mod(positions,1)
    print("done")

    Nfolded = np.any(folded_positions != positions,axis=1).sum()
    print("\n\tNumber of structures that have been unfolded: {:d}".format(Nfolded))

    #------------------#
    print("\tSetting unfolded fractional/scaled coordinates ... ", end="")
    folded_positions = folded_positions.reshape((N,*shape))
    for n in range(len(atoms)):
        atoms[n].set_scaled_positions(folded_positions[n,:,:])
    print("done")

    #------------------#
    for n in range(len(atoms)):
        pos = atoms[n].get_scaled_positions(wrap=False)
        if np.any(np.abs(pos)>1):
            raise ValueError('coding error')

    #------------------#
    # bool2str = lambda value: "true" if value else "false"
    # positions = positions.reshape((N,-1))
    # a = positions.reshape((N,-1))
    # b = folded_positions.reshape((N,-1))
    # diff = np.diff(a-b,axis=0)
    # modified = np.square(diff).sum(axis=1) > 0.1
    # index = np.where(modified == True)[0]
    # unfolded = len(index) > 0 
    # tf = bool2str(unfolded)
    # print("\n\tNumber of structures that have been unfolded: {:d}".format(len(index)))
    # print("\t{:s}: this estimation is wrong".format(warning))
    # print("\n\tAt least one coordinate has been unfolded: {:s}".format(tf))
    # if unfolded:
    #     modified = (np.sqrt(np.square(a-b).sum(axis=1)) > 0.1) # whether they have been modified
    #     index = np.where(modified == True)[0]

    #------------------#
    print("\n\tWriting unfolded structures to output file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, atoms, format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

#---------------------------------------#
if __name__ == "__main__":
    main()