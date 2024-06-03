#!/usr/bin/env python
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt, warning
from eslib.classes.physical_tensor import PhysicalTensor, corrected_sqrt
from eslib.output import output_folder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from eslib.plot import square_plot, plot_bisector, hzero
from icecream import ic
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Compare two sets of normal modes."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a", "--modes_A", type=str, **argv, required=True , help="normal modes A")
    parser.add_argument("-b", "--modes_B", type=str, **argv, required=True , help="normal modes B")
    parser.add_argument("-d" , "--dof"   , type=int, **argv, required=False, help="dof to be skipped (default: %(default)s)", default=3)
    parser.add_argument("-o", "--output" , type=str, **argv, required=True , help="output folder")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    print("\n\tReading normal modes A from file '{:s}' ... ".format(args.modes_A),end="")
    A = NormalModes.from_pickle(args.modes_A)
    print("done")

    #---------------------------------------#
    print("\tReading normal modes A from file '{:s}' ... ".format(args.modes_B),end="")
    B = NormalModes.from_pickle(args.modes_B)
    print("done")

    #---------------------------------------#
    output_folder(args.output)

    #---------------------------------------#
    Amodes = A.mode.copy()
    Bmodes = B.mode.copy()

    Amodes = PhysicalTensor(Amodes).rename({'mode': 'mode-a'})  
    Bmodes = PhysicalTensor(Bmodes).rename({'mode': 'mode-b'})   
    prod =  Amodes.cdot(Bmodes,"dof").to_numpy()[args.dof:,args.dof:]
    cos = np.arccos(prod)*180/np.pi
    cos = np.asarray([ 90 - abs(c - 90) if c > 90 else c for c in cos.flatten() ]).reshape(cos.shape)

    if np.any(np.abs(prod)>1):
        print("\t{:s}: values with modulus greater than 1.".format(warning))

    #---------------------------------------#    
    file = "{:s}/prod.txt".format(args.output)
    np.savetxt(file,prod)

    file = "{:s}/cos.txt".format(args.output)
    np.savetxt(file,cos)

    #---------------------------------------#
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue to white to red
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    plt.figure(figsize=(8, 6))
    sns.heatmap(prod, cmap=cmap, annot=False)
    plt.savefig("{:s}/prod.pdf".format(args.output))

    #
    colors = [(1, 1, 1),(0, 0, 0)]  # Blue to white to red
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cos, cmap=cmap, annot=False)
    plt.savefig("{:s}/cos.pdf".format(args.output))

    #---------------------------------------#
    # Example data
    factor = convert(1,"frequency","atomic_unit","thz")
    array1 = corrected_sqrt(A.get("eigval"))*factor
    array2 = corrected_sqrt(B.get("eigval"))*factor

    # Plotting the correlation heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(6, 6))
    ax1.scatter(array1,array2,color="blue",s=5)
    ax1.grid()
    ax1.set_xlabel("A freq. [THz]")
    ax1.set_ylabel("B freq. [THz]")
    square_plot(ax1)
    plot_bisector(ax1)

    width = (array1.max() - array1.min())/ (5*len(array1))
    y = np.abs(array1-array2)
    ax2.bar(array1,y,width=width,color="blue")
    ax2.grid()
    ax2.set_xlabel("A freq. [THz]")
    ax2.set_ylabel("$\\Delta$ freq. [THz]",color="blue")
    ax2.set_ylim(0,y.max()*1.1)
    hzero(ax2)

    # Create a twin axis for ax2
    ax3 = ax2.twinx()
    ax3.set_ylabel("perc. err.",color="red")  # Label for the right y-axis
    y = 100 * np.abs((array1-array2)/array1)
    ax3.bar(array1,y,width=width,color="red")

    ax3.set_ylim(0,y.max()*1.1)
    # hzero(ax3)

    plt.tight_layout()
    plt.savefig("{:s}/corr-eigval.pdf".format(args.output))


    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
