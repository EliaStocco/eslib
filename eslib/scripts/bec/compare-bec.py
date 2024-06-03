#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
import matplotlib.pyplot as plt

#---------------------------------------#
# Description of the script's purpose
description = "Compare BECs"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-a", "--bec_a", **argv, required=True, type=str, help="*.txt input file with the first BECs")
    parser.add_argument("-b", "--bec_b", **argv, required=True, type=str, help="*.txt input file with the second BECs")
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output file (default: %(default)s)", default='bec.pdf')
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    shrink = 1

    Za = np.loadtxt(args.bec_a)
    Zb = np.loadtxt(args.bec_b)
    dZ = np.abs(Za - Zb)
    Zd = dZ / np.abs(Za)
    
    # Determine global vmin and vmax for consistent color scale
    global_vmin = min(np.min(Za), np.min(Zb))
    global_vmax = max(np.max(Za), np.max(Zb))

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Plot Za
    im0 = axes[0].imshow(Za, cmap='coolwarm', origin='upper', aspect='auto', vmin=global_vmin, vmax=global_vmax)
    axes[0].set_title('ref.')
    axes[0].set_aspect('equal')
    axes[0].set_xticks(np.arange(Za.shape[1]))  # Set x ticks
    axes[0].set_xticklabels(['x', 'y', 'z'])     # Set x tick labels
    for i in range(Za.shape[0]):
        for j in range(Za.shape[1]):
            axes[0].text(j, i, '{:.2f}'.format(Za[i, j]), ha='center', va='center', color='black')

    # Plot Zb
    im1 = axes[1].imshow(Zb, cmap='coolwarm', origin='upper', aspect='auto', vmin=global_vmin, vmax=global_vmax)
    axes[1].set_title('pred.')
    axes[1].set_aspect('equal')
    axes[1].set_xticks(np.arange(Zb.shape[1]))  # Set x ticks
    axes[1].set_xticklabels(['x', 'y', 'z'])     # Set x tick labels
    for i in range(Zb.shape[0]):
        for j in range(Zb.shape[1]):
            axes[1].text(j, i, '{:.2f}'.format(Zb[i, j]), ha='center', va='center', color='black')

    # Plot dZ
    im2 = axes[2].imshow(dZ, cmap='Blues', origin='upper', aspect='auto')
    axes[2].set_title('abs. diff.')
    axes[2].set_aspect('equal')
    for i in range(dZ.shape[0]):
        for j in range(dZ.shape[1]):
            axes[2].text(j, i, '{:.2f}'.format(dZ[i, j]), ha='center', va='center', color='black')

    # Plot Zd/abs(Za)
    im3 = axes[3].imshow(Zd, cmap='Blues', origin='upper', aspect='auto', vmin=0, vmax=1)
    axes[3].set_title('rel. diff.')
    axes[3].set_aspect('equal')
    for i in range(Zd.shape[0]):
        for j in range(Zd.shape[1]):
            axes[3].text(j, i, '{:.2f}'.format(Zd[i, j]), ha='center', va='center', color='black')

    # Add a common colorbar for Za and Zb
    cbar = fig.colorbar(im1, ax=[axes[0], axes[1]], shrink=shrink)

    # Add a separate colorbar for dZ and Zd
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=shrink)
    cbar3 = fig.colorbar(im3, ax=axes[3], shrink=shrink)

    # Save the plot
    plt.savefig(args.output)

#---------------------------------------#
if __name__ == "__main__":
    main()
