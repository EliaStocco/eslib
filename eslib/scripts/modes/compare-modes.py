#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from icecream import ic
from matplotlib.colors import LinearSegmentedColormap

from eslib.classes.normal_modes import NormalModes
from eslib.classes.physical_tensor import PhysicalTensor, corrected_sqrt
from eslib.formatting import esfmt, warning
from eslib.output import output_folder
from eslib.plot import hzero, plot_bisector, square_plot
from eslib.tools import convert

# Set the backend to QtAgg
matplotlib.use('Agg')

# Enable interactive mode
plt.ion()

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
    Amodes = A.mode.copy().real
    Bmodes = B.mode.copy().real

    Amodes = PhysicalTensor(Amodes).rename({'mode': 'mode-a'})  
    Bmodes = PhysicalTensor(Bmodes).rename({'mode': 'mode-b'})   
    prod =  np.abs(Amodes.cdot(Bmodes,"dof").to_numpy()[args.dof:,args.dof:])
    cos = np.arccos(prod)*180/np.pi
    cos = np.abs(np.asarray([ 90 - abs(c - 90) if c > 90 else c for c in cos.flatten() ]).reshape(cos.shape))

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

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    plt.title("Scalar product between normal modes: $|{\\rm N}_i^A\\cdot {\\rm N}_j^B|$")
    sns.heatmap(prod, cmap='viridis', annot=False)
    # im = ax.imshow(prod, cmap='viridis')
    # ax.set_aspect('equal')
    # # square_plot(ax)
    # plt.colorbar(im, ax=ax, shrink=1)
    plt.savefig("{:s}/prod.pdf".format(args.output))

    #
    colors = [(1, 1, 1),(0, 0, 0)]  # Blue to white to red
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    plt.figure(figsize=(6, 6))
    plt.title("Angle [deg] between normal modes: arccos $\\left({\\rm N}_i^A\\cdot {\\rm N}_j^B\\right)$")
    sns.heatmap(cos, cmap='viridis', annot=False)
    plt.savefig("{:s}/cos.pdf".format(args.output))

    #---------------------------------------#
    # Example data
    factor = convert(1,"frequency","atomic_unit","thz")
    array1 = corrected_sqrt(A.get("eigval"))*factor
    array2 = corrected_sqrt(B.get("eigval"))*factor

    ii = np.logical_and(array1 > 0,array2 > 0 )
    array1 = array1[ii]
    array2 = array2[ii]

    # Plotting the correlation heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True,figsize=(6, 10))
    ax1.scatter(array1,array2,color="blue",s=5)
    # ax1.grid()
    ax1.set_xlabel("A freq. [THz]")
    ax1.set_ylabel("B freq. [THz]")
    square_plot(ax1)
    plot_bisector(ax1)

    width = (array1.max() - array1.min())/ (5*len(array1))
    y = np.abs(array1-array2)
    ax2.bar(array1,y,width=width,color="blue")
    # ax2.grid()
    ax2.set_ylabel("$\\Delta$ freq. [THz]",color="blue")
    ax2.set_ylim(0,y.max()*1.1)
    hzero(ax2)

    # Create a twin axis for ax2
    # ax3 = ax2.twinx()
    ax3.set_ylabel("perc. err.",color="red")  # Label for the right y-axis
    y = 100 * np.abs((array1-array2)/array1)
    ax3.bar(array1,y,width=width,color="red")
    hzero(ax3)
    ax3.set_ylim(0,y.max()*1.1)
    # hzero(ax3)
    ax3.set_xlabel("A freq. [THz]")

    # xlim = ax3.get_xlim()
    

    for ax in [ax1, ax2, ax3]:
        ax.grid()

    plt.tight_layout()
    plt.savefig("{:s}/corr-eigval.pdf".format(args.output))


    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
