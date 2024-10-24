#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from ase.geometry.rdf import get_rdf
from matscipy.neighbours import neighbour_list

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import itype, nslist

#---------------------------------------#
description = "Compute and plot the Radial Distribution Function (RDF)."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, type=str  , help="input file")
    parser.add_argument("-if", "--input_format", **argv, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-in", "--index"       , **argv,required=False, type=itype   , help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-n" , "--nbins"       , **argv, type=int  , help="Number of bins to divide RDF (default: %(default)s)", default=100)
    parser.add_argument("-m" , "--min"         , **argv, type=float, help="minimum distance in the plot (default: %(default)s)", default=2)
    parser.add_argument("-r" , "--rmax"        , **argv, type=float, help="Maximum distance of RDF (default: %(default)s)", default=5.)
    parser.add_argument("-e" , "--elements"    , **argv, type=nslist, help="elements (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"      , **argv, type=str  , help="output file (default: %(default)s)", default="rdf.csv")
    parser.add_argument("-p" , "--plot"        , **argv, type=str  , help="plot (default: %(default)s)", default=None)
    return parser 

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    args.rmax = float(args.rmax)

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format,index=args.index)
    print("done")
    N = len(trajectory)
    print("\tn. of atomic structures: {:d}".format(N))

    if np.asarray(args.elements).ndim == 0:
        raise ValueError("elements must be a list or tuple of length 1 or 2")
    elif np.asarray(args.elements).ndim == 1:
        args.elements = [args.elements]
    elif np.asarray(args.elements).ndim > 2:
        raise ValueError("elements must be a list or tuple of length 1 or 2")
    elements_list = [None]*len(args.elements)
    for n,elements in enumerate(args.elements):
        elements_list[n] =  [atomic_numbers[symbol] for symbol in elements ]


    # print("\tPreparing analysis ... ".format(args.input), end="")
    # analysis = Analysis(atoms)
    # print("done\n")

    Nl = len(elements_list)
    print("\tComputing RDF: ")
    # rdf = np.zeros((Nl,N,))
    for n,atoms in enumerate(trajectory):
        print("\t{:d}/{:d} ... ".format(n+1,N), end="\r")
        first,second,distance = neighbour_list(quantities="ijd",atoms=atoms,cutoff=args.rmax)
        distance_matrix = np.full((len(atoms),len(atoms)),1000000.0) # float!!
        distance_matrix[first,second] = distance
        distance_matrix[second,first] = distance
        # for nn,(ii,jj) in enumerate(zip(first,second)):
        #     distance_matrix[ii, jj] = distance[nn]
        #     distance_matrix[jj, ii] = distance[nn]
        np.fill_diagonal(distance_matrix, 0.0)
        # distance_matrix[second, first] = distance  # Ensure symmetry
        if n == 0:
            for i,elements in enumerate(elements_list):
                tmp,dist = get_rdf(atoms=atoms,nbins=args.nbins, rmax=args.rmax,elements=elements,no_dists=False,distance_matrix=distance_matrix)
                if i == 0 :
                    rdf = np.zeros((Nl,N,len(tmp))) 
                rdf[i,0,:] = tmp
        else:
            for i,elements in enumerate(elements_list):
                rdf[i,n,:] = get_rdf(atoms=atoms,nbins=args.nbins, rmax=args.rmax,elements=elements,no_dists=True,distance_matrix=distance_matrix)
        pass

    print("\n\tRDF.shape:",rdf.shape)
    
    print()
    for i,elements in enumerate(elements_list):
        E0 = next((k for k, v in atomic_numbers.items() if v == elements[0]), None)
        E1 = next((k for k, v in atomic_numbers.items() if v == elements[1]), None)
        filename, extension = os.path.splitext(args.output)
        file = f"{filename}.{E0}-{E1}{extension}"
        print("\tSaving RDF for {:s}-{:s} to file '{:s}' ... ".format(E0,E1,file), end="")
        df = pd.DataFrame()
        df["distance"] = dist
        df["rdf"] = rdf[i,:,:].mean(axis=0)
        df["std"] = rdf[i,:,:].std(axis=0)
        df["err"] = df["std"]/np.sqrt(rdf.shape[1])
        df.to_csv(file,index=False)
        print("done")
    print("")

    #---------------------------------------#
    # plot
    if args.plot is not None:
        if np.asarray(args.elements).ndim == 2:
            raise ValueError("plot not implemented yet")
        
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

