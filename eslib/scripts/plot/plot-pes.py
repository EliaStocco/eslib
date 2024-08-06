#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.formatting import esfmt
from classes.atomic_structures import AtomicStructures
from eslib.input import ilist
import pandas as pd
from itertools import product

#---------------------------------------#
# Description of the script's purpose
description = "Plot a time series of a 'info' of a 'extxyz' file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input"  , type=str  , **argv, required=True , help='input extxyz file')
    parser.add_argument("-e", "--energy" , type=str  , **argv, required=False , help="info keyword of the energy (default: %(default)s)", default='harmonic-potential-energy')
    parser.add_argument("-n", "--indices", type=str  , **argv, required=False, help="info keyword of the 2D indices (default: %(default)s)", default=None)
    parser.add_argument("-s", "--shape"  , type=ilist, **argv, required=False, help="shape of the PES (default: %(default)s)", default=None)
    parser.add_argument("-o", "--output" , type=str  , **argv, required=False, help="pdf output file (default: %(default)s)", default='pes.pdf')
    return parser# .parse_args()

#---------------------------------------#
def plot_pes(energy, indices, file):
    x = indices[:, 0]
    y = indices[:, 1]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the triangular mesh with color shading
    tripcolor = plt.tripcolor(x, y, energy, shading='gouraud')
    
    # Add level curves
    levels = np.linspace(np.min(energy), np.max(energy), 10)  # Adjust the number of levels as needed
    plt.tricontour(x, y, energy, levels=levels, colors='white', linestyles='dashed')
    
    # Add colorbar for the tripcolor plot only
    plt.colorbar(tripcolor, label='Energy')
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential Energy Surface')
    
    # Save the plot to file
    plt.savefig(file)
    plt.close()  # Close the plot to avoid displaying it in the notebook (if not desired)


#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    assert args.indices is not None or args.shape is not None, "--indices and --shape can not be both 'None'"
    assert len(args.shape) == 2, "--shape must contain two numbers."

    #------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input)
    print("done")

    #------------------#
    # data
    print("\n\tExtracting '{:s}' from the atomic structures ... ".format(args.energy), end="")
    energy = atoms.get_info(args.energy)
    print("done")
    print("\t'{:s}' shape: ".format(args.energy),energy.shape)

    assert energy.shape == (len(atoms),)

    #------------------#
    if args.indices is None:
        assert len(atoms) == args.shape[0]*args.shape[1]
        
        instructions = pd.DataFrame(columns=["mode","start","end","N"],index=[0,1])

        instructions.at[0,"mode"]  = 0
        instructions.at[0,"start"] = 0
        instructions.at[0,"end"]   = 1
        instructions.at[0,"N"]     = args.shape[0]-1

        instructions.at[1,"mode"]  = 1
        instructions.at[1,"start"] = 0
        instructions.at[1,"end"]   = 1
        instructions.at[1,"N"]     = args.shape[1]-1

        #------------------#
        print("\tPreparing displacement along single modes:")
        lists = [None]*len(instructions)
        for n,row in instructions.iterrows():
            print("\t\tmode {:3d}: ".format(int(row['mode'])),end="")
            lists[n]= np.linspace(row['start'],row['end'],int(row['N']+1),endpoint=True)
            print(lists[n])

        displacements = np.asarray(list(product(*lists)))
        print(displacements.shape)

        atoms.set("indices",displacements,"info")
        args.indices = "indices"
        
    #------------------#
    print("\n\tExtracting '{:s}' from the atomic structures ... ".format(args.indices), end="")
    indices = atoms.get(args.indices)
    print("done")
    print("\t'{:s}' shape: ".format(args.indices),indices.shape)

    assert indices.shape == (len(atoms),2)

    #------------------#
    print("\n\tSaving the PES plot to file ... ".format(args.output), end="")
    plot_pes(energy,indices,args.output)
    print("done")
    

if __name__ == "__main__":
    main()
