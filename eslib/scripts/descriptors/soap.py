#!/usr/bin/env python
import numpy as np
from ase import Atoms
from typing import List
from dscribe.descriptors import SOAP
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
from eslib.classes.append import AppendableList

#---------------------------------------#
# Description of the script's purpose
description = "Compute the SOAP descriptors for a bunch of atomic structures, with optional chunking to limit memory use."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--input"       , type=str, required=True , **argv, help="input file [au]")
    parser.add_argument("-if", "--input_format", type=str, required=False, **argv, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"      , type=str, required=False, **argv, help="output file with SOAP descriptors (default: %(default)s)", default="soap.npy")
    parser.add_argument("-N" , "--chunk_size"  , type=int, required=False, **argv, help="number of structures per chunk (default: process all at once)", default=100)
    return parser

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #-------------------#
    print(f"\n\tReading structures from file '{args.input}' ... ", end="")
    structures: List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    n_structures = len(structures)
    print(f"\tn. of structures: {n_structures}")
    Natoms = structures.call(lambda x: x.get_global_number_of_atoms())
    print(f"\tn. of atoms: {np.unique(Natoms)}")
    species = structures.get_chemical_symbols(unique=True)
    print(f"\tspecies: {species}")

    #-------------------#
    print("\n\tPreparing SOAP object ... ", end="")
    soap = SOAP(
        species=species,
        r_cut=5.0,
        n_max=8,
        l_max=6,
        sigma=0.3,
        periodic=False,
        sparse=False,
    )
    print("done")

    #-------------------#
    if args.chunk_size is None or args.chunk_size <= 0:
        chunk_size = n_structures
    else:
        chunk_size = args.chunk_size

    print("\tComputing SOAP descriptors: ")

    X_list = AppendableList()
    for i in range(0, n_structures, chunk_size):
        chunk = structures[i : i + chunk_size]
        print(f"\t\tprocessing structures {i}-{min(i+chunk_size, n_structures)} ... ", end="")
        X_chunk = soap.create(chunk)
        X_list.append(X_chunk)
        print("done")

    X = np.vstack(X_list.finalize())
    print("\n\tAll chunks processed.")
    print(f"\tSOAP (atom-wise) feature matrix shape: {X.shape}")

    # Global average over atoms
    Xglobal = X.mean(axis=1)
    print(f"\tSOAP (global) feature matrix shape: {Xglobal.shape}")

    #-------------------#
    print(f"\n\tSaving SOAP descriptors to file '{args.output}' ... ", end="")
    if str(args.output).endswith("npy"):
        np.save(args.output, Xglobal.T)
    elif str(args.output).endswith("txt"):
        header = "Rows: SOAP descriptor\nCols: structures"
        np.savetxt(args.output, Xglobal.T, fmt=float_format, header=header)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()
