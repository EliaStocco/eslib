#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
import matplotlib.pyplot as plt
from eslib.input import str2bool, ilist
from eslib.classes.bec import bec

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
    parser.add_argument("-sn", "--show_numbers", **argv, required=False, type=str2bool, help="show numbers (default: %(default)s)", default=True)
    parser.add_argument("-fs", "--figsize", **argv, required=False, type=ilist, help="figsize (default: %(default)s)", default=[15,5])
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output file (default: %(default)s)", default='bec.pdf')
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    shrink = 1

    Za = bec.from_file(file=args.bec_a)
    Zb = bec.from_file(file=args.bec_b)
    dZ = Za - Zb
    # Zd = dZ / np.abs(Za)
    Zd = dZ.relative_to(Za)
    dZ = np.abs(dZ[0,:,:])
    Za = Za[0,:,:]
    Zb = Zb[0,:,:]
    Zd = 100*np.abs(Zd[0,:,:])

    print("\n\tPercentage difference:")
    print("\t\t max: {:.4f}".format(Zd.max()))
    print("\t\tmean: {:.4f}".format(Zd.mean()))
    print("\t\t std: {:.4f}".format(Zd.std()))

    Natoms = int(Zd.shape[0]/3)    
    atomic = Zd.to_numpy().reshape((Natoms,3,3))
    print("\n\tPercentage difference (on-diagonal):")
    
    ondiag = np.asarray([ np.diag(a) for a in atomic])
    print("\t\t max: {:.4f}".format(ondiag.max()))
    print("\t\tmean: {:.4f}".format(ondiag.mean()))
    print("\t\t std: {:.4f}".format(ondiag.std()))

    
    print("\n\tPercentage difference (off-diagonal):")
    mask = ~np.eye(3, dtype=bool)
    Natoms = int(Zd.shape[0]/3)
    offdiag = np.asarray([ a[mask].flatten() for a in atomic])
    print("\t\t max: {:.4f}".format(offdiag.max()))
    print("\t\tmean: {:.4f}".format(offdiag.mean()))
    print("\t\t std: {:.4f}".format(offdiag.std()))

    
    # Determine global vmin and vmax for consistent color scale
    global_vmin = min(np.min(Za), np.min(Zb))
    global_vmax = max(np.max(Za), np.max(Zb))

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=tuple(args.figsize))

    # Plot Za
    im0 = axes[0].imshow(Za, cmap='coolwarm', origin='upper', aspect='auto', vmin=global_vmin, vmax=global_vmax)
    axes[0].set_title('ref.')
    axes[0].set_aspect('equal')
    axes[0].set_xticks(np.arange(Za.shape[1]))  # Set x ticks
    axes[0].set_xticklabels(['x', 'y', 'z'])     # Set x tick labels
    if args.show_numbers:
        for i in range(Za.shape[0]):
            for j in range(Za.shape[1]):
                axes[0].text(j, i, '{:.2f}'.format(Za[i, j]), ha='center', va='center', color='black')

    # Plot Zb
    im1 = axes[1].imshow(Zb, cmap='coolwarm', origin='upper', aspect='auto', vmin=global_vmin, vmax=global_vmax)
    axes[1].set_title('pred.')
    axes[1].set_aspect('equal')
    axes[1].set_xticks(np.arange(Zb.shape[1]))  # Set x ticks
    axes[1].set_xticklabels(['x', 'y', 'z'])     # Set x tick labels
    if args.show_numbers:
        for i in range(Zb.shape[0]):
            for j in range(Zb.shape[1]):
                axes[1].text(j, i, '{:.2f}'.format(Zb[i, j]), ha='center', va='center', color='black')

    # Plot dZ
    im2 = axes[2].imshow(dZ, cmap='Blues', origin='upper', aspect='auto',vmin=dZ.min(), vmax=dZ.max())
    axes[2].set_title('abs. diff.')
    axes[2].set_aspect('equal')
    axes[2].set_xticks(np.arange(dZ.shape[1]))  # Set x ticks
    axes[2].set_xticklabels(['x', 'y', 'z'])     # Set x tick labels
    if args.show_numbers:
        for i in range(dZ.shape[0]):
            for j in range(dZ.shape[1]):
                axes[2].text(j, i, '{:.2f}'.format(dZ[i, j]), ha='center', va='center', color='black')

    # Plot Zd/abs(Za)
    im3 = axes[3].imshow(Zd, cmap='Blues', origin='upper', aspect='auto', vmin=Zd.min(), vmax=Zd.max())
    axes[3].set_title('perc. diff.')
    axes[3].set_aspect('equal')
    axes[3].set_xticks(np.arange(Zd.shape[1]))  # Set x ticks
    axes[3].set_xticklabels(['x', 'y', 'z'])     # Set x tick labels
    if args.show_numbers:
        for i in range(Zd.shape[0]):
            for j in range(Zd.shape[1]):
                axes[3].text(j, i, '{:.2f}'.format(Zd[i, j]), ha='center', va='center', color='black')

    # Add a common colorbar for Za and Zb
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=shrink)
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=shrink)
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=shrink)
    cbar3 = fig.colorbar(im3, ax=axes[3], shrink=shrink)

    # Save the plot
    plt.tight_layout()
    plt.savefig(args.output)

#---------------------------------------#
if __name__ == "__main__":
    main()
