#!/usr/bin/env python
import matplotlib.pyplot as plt
from eslib.plot import plot_bisector
from eslib.formatting import esfmt
from classes.atomic_structures import AtomicStructures
from eslib.physics import compute_dipole_quanta
from eslib.plot import square_plot
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Plot the correlation between two info of a extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input"     , type=str, **argv, required=True , help='input extxyz file')
    parser.add_argument("-a" , "--dipole_A", type=str, **argv, required=True , help="dipole A keyword")
    parser.add_argument("-b" , "--dipole_B", type=str, **argv, required=True , help="dipole B keyword")
    parser.add_argument("-is" , "--isolated", type=str2bool, **argv, required=True , help="isolated (default: %(default)s)", default=False)
    parser.add_argument("-s", "--size"  , type=float, **argv, required=False, help="point size (default: %(default)s)", default=2)
    parser.add_argument("-o", "--output"    , type=str, **argv, required=False, help="output file (default: %(default)s)", default='corr.pdf')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input)
    print("done")

    #------------------#
    if not args.isolated:
        try:
            quanta_A = "quanta-{:s}".format(args.dipole_A)
            quanta_B = "quanta-{:s}".format(args.dipole_A)
            print("\tComputing dipole quanta for '{:s}' ... ".format(args.dipole_A), end="")
            atoms: AtomicStructures = compute_dipole_quanta(atoms,args.dipole_A,quanta_A)[0]
            print("done")

            print("\tComputing dipole quanta for '{:s}' ... ".format(args.dipole_B), end="")
            atoms:AtomicStructures = compute_dipole_quanta(atoms,args.dipole_B,quanta_B)[0]
            print("done")
        except:
            quanta_A = args.dipole_A
            quanta_B = args.dipole_B
    else:
        quanta_A = args.dipole_A
        quanta_B = args.dipole_B

    #------------------#
    A = atoms.get(quanta_A)
    B = atoms.get(quanta_B)
    assert A.shape == B.shape

    #------------------#
    fig,ax = plt.subplots(figsize=(4.5,4))

    labels = ["$\\mathbf{\\mu}_1$","$\\mathbf{\\mu}_2$","$\\mathbf{\\mu}_3$"]
    for n in range(3):
        ax.scatter(A[:,n],B[:,n],label=labels[n],s=args.size)
        plot_bisector(ax,argv={"linewidth":0.5})
        ax.grid()
    
    ax.legend(loc="upper left",title="dipole:",facecolor='white', framealpha=1,edgecolor="black")
    ax.set_xlabel(args.dipole_A)
    ax.set_ylabel(args.dipole_B)
    square_plot(ax)

    plt.tight_layout()
    print("\tSaving plot to file '{:s}' ... ".format(args.output),end="")
    plt.savefig(args.output)
    print("done")


if __name__ == "__main__":
    main()
