#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.formatting import esfmt
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Plot a time series of a 'info' of a 'extxyz' file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input"  , type=str, **argv, required=True , help='input extxyz file')
    parser.add_argument("-e", "--energy" , type=str, **argv, required=False , help="info keyword of the energy (default: 'harmonic-potential-energy')", default='harmonic-potential-energy')
    parser.add_argument("-n", "--indices", type=str, **argv, required=False, help="info keyword of the 2D indices (default: 'displacements')", default='displacements')
    parser.add_argument("-o", "--output" , type=str, **argv, required=False, help="pdf output file (default: 'pes.pdf')", default='pes.pdf')
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
    print("\n\tExtracting '{:s}' from the atomic structures ... ".format(args.indices), end="")
    indices = atoms.get_info(args.indices)
    print("done")
    print("\t'{:s}' shape: ".format(args.indices),indices.shape)

    assert indices.shape == (len(atoms),2)

    #------------------#
    print("\n\tSaving the PES plot to file ... ".format(args.output), end="")
    plot_pes(energy,indices,args.output)
    print("done")
    

if __name__ == "__main__":
    main()
