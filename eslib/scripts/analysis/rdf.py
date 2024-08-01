#!/usr/bin/env python
import numpy as np
import pandas as pd
from ase.geometry.rdf import get_rdf
from ase.geometry.analysis import Analysis
from classes import trajectory
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import slist
import matplotlib.pyplot as plt
from ase.data import chemical_symbols, atomic_numbers

#---------------------------------------#
description = "Compute and plot the Radial Distribution Function (RDF)."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, type=str  , help="input file")
    parser.add_argument("-if", "--input_format", **argv, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--nbins"       , **argv, type=int  , help="Number of bins to divide RDF (default: %(default)s)", default=100)
    parser.add_argument("-m" , "--min"         , **argv, type=float, help="minimu distance in the plot (default: %(default)s)", default=2)
    parser.add_argument("-r" , "--rmax"        , **argv, type=float, help="Maximum distance of RDF (default: %(default)s)", default=5)
    parser.add_argument("-e" , "--elements"    , **argv, type=slist, help="elements (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"      , **argv, type=str  , help="output file (default: %(default)s)", default="rdf.csv")
    parser.add_argument("-p" , "--plot"        , **argv, type=str  , help="plot (default: %(default)s)", default=None)
    return parser 

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done\n")
    N = len(trajectory)
    print("\tn. of atomic structures: {:d}".format(N))

    elements =  [ atomic_numbers[symbol] for symbol in args.elements]

    # print("\tPreparing analysis ... ".format(args.input), end="")
    # analysis = Analysis(atoms)
    # print("done\n")

    print("\tComputing RDF: ")
    rdf = np.zeros((N,))
    for n,atoms in enumerate(trajectory):
        print("\t{:d}/{:d} ... ".format(n+1,N), end="\r")
        if n == 0:
            tmp,dist = get_rdf(atoms=atoms,nbins=args.nbins, rmax=args.rmax,elements=elements,no_dists=False)
            rdf = np.zeros((len(trajectory),len(tmp)))
            rdf[0,:] = tmp
        else:
            rdf[n,:] = get_rdf(atoms=atoms,nbins=args.nbins, rmax=args.rmax,elements=elements,no_dists=True)
        pass
    # print("done\n")

    print("\n\tRDF.shape:",rdf.shape)

    print("\tSaving RDF to file '{:s}' ... ".format(args.output), end="")
    df = pd.DataFrame()
    df["distance"] = dist
    df["rdf"] = rdf.mean(axis=0)
    df["std"] = rdf.std(axis=0)
    df["err"] = df["std"]/np.sqrt(rdf.shape[0])
    df.to_csv(args.output,index=False)
    print("done\n")

    #---------------------------------------#
    # plot
    if args.plot is not None:
        print("\tPreparing plot ... ".format(args.input), end="")
        plt.figure(figsize=(6,4))
        plt.plot(df["distance"],df["rdf"],color="blue")
        plt.fill_between(df["distance"],df["rdf"]-df["err"],df["rdf"]+df["err"],alpha=0.5,color="grey")
        plt.xlabel("distance $[\\AA]$")
        plt.ylabel("RDF")
        plt.xlim(args.min,args.rmax)
        plt.ylim(0,None)
        plt.grid()
        # plt.legend()
        plt.tight_layout()        
        print("done\n")

        print("\tSaving plot to file '{:s}' ... ".format(args.plot), end="")
        plt.savefig(args.plot)
        plt.close()
        print("done\n")
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()

