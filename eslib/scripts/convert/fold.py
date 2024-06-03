#!/usr/bin/env python
import argparse
import numpy as np
from eslib.formatting import esfmt
from eslib.tools import cart2frac, frac2cart
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
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m"  , "--method"       ,   **argv,type=str, help="method (default: 'eslib')", default='eslib', choices=['ase','eslib'])
    parser.add_argument("-o"  , "--output"       ,   **argv,type=str, help="output file")
    parser.add_argument("-of" , "--output_format",   **argv,type=str, help="output file format (default: %(default)s)", default=None)
    # options = parser.parse_args()
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from input file '{:s}' ... ".format(args.input), end="")
    # atoms = read(args.input,index=":",format=args.input_format)
    atoms:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    N = len(atoms)
    shape = atoms[0].positions.shape
    print("\t{:s}: {:d}".format("Number of atomic structures",N))
    print("\t{:s}: ({:d},{:d})".format("Positions shape of each structure",shape[0],shape[1]))

    #------------------#
    if args.method == 'ase':

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
            
    elif args.method == 'eslib':

        print("\n\tComputing fractional/scaled coordinates into the primitive cell ... ", end="")
        
        frac_positions = np.full((N,*shape),np.nan)
        for n in range(len(atoms)):
            frac_positions[n,:,:] = cart2frac(atoms[n].get_cell(),atoms[n].positions)
        print("done")

        print("\tFolding fractional/scaled coordinates ... ", end="")
        old_frac = frac_positions.copy()
        frac_positions = np.mod(frac_positions,1)
        print("done")
        Nfolded = np.any( (frac_positions != old_frac ).reshape((len(atoms),-1)) , axis=1).sum()
        print("\n\tNumber of structures that have been folded: {:d}".format(Nfolded))

        #------------------#
        print("\tSetting unfolded fractional/scaled coordinates ... ", end="")
        for n in range(len(atoms)):
            atoms[n].positions = frac2cart(atoms[n].get_cell(),frac_positions[n])
        print("done")

        #------------------#
        for n in range(len(atoms)):
            pos = atoms[n].get_scaled_positions(wrap=False)
            if np.any(np.abs(pos)>1):
                raise ValueError('coding error')

    #------------------#
    print("\n\tWriting unfolded structures to output file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(file=args.output,format=args.output_format)
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
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/dipole/eval-dipole-model.py",
#             "cwd" : "/home/stoccoel/google-personal/works/BaTiO3/data/elia/pes/",
#             "console": "integratedTerminal",
#             "args" : ["-i", "Training_set.xyz", "-o", "structures.extxyz", "-of", "extxyz", "-if", "extxyz"],
#             "justMyCode": false,
#         }
#     ]
# }