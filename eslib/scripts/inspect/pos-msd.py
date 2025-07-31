#!/usr/bin/env python
import numpy as np
import pandas as pd
from ase import Atoms
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist
from eslib.plot import hzero
matplotlib.use('Agg')

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Mean Squared Displacement of the positions of a trajectory w.r.t. a rerefence structure."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-r" , "--ref"          , **argv, type=str, required=False, help="reference input file [extxyz] (default: %(default)s)", default=None)
    parser.add_argument("-rf", "--ref_format"   , **argv, type=str, required=False, help="reference input file format (default: %(default)s)" , default=None)
    parser.add_argument("-oi" , "--output_info" , **argv, type=str, required=False, help="output txt file with the MSD of each atom (default: %(default)s)", default="MSD.array.txt")
    parser.add_argument("-oa" , "--output_array", **argv, type=str, required=False, help="output txt file with the global MSD (default: %(default)s)", default="MSD.info.txt")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    # atomic structures
    print(f"\n\tReading atomic structures from file '{args.input}' ... ", end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)
    Natoms = structures.num_atoms()
    print("\tn. of atoms: ",Natoms)
    
    #------------------#
    # atomic structures
    if args.ref is not None:
        print(f"\n\tReading reference structure from file '{args.ref}' ... ", end="")
        reference:Atoms = AtomicStructures.from_file(file=args.ref,format=args.ref_format,index=0)[0]
        print("done")
        print("\tn. of atoms: ",reference.get_global_number_of_atoms())
    else:
        reference:Atoms = structures[0].copy()
        
    #------------------#
    print("\n\tExtracting positions ... ",end="")
    pos = structures.get("positions")
    ref = reference.get_positions()[None,:,:]
    print("done")
    print("\tpos.shape: ",pos.shape)
    print("\tref.shape: ",ref.shape)
    
    print("\n\tComputing MPSD ... ",end="")
    
    return

if __name__ == "__main__":
    main()

