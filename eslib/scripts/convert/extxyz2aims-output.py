#!/usr/bin/env python
import numpy as np
import xml.etree.ElementTree as ET
import ast
from typing import Tuple, List
from ase import Atoms
from ase.io import write
from eslib.tools import convert
import os
from eslib.formatting import esfmt
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Convert an i-PI checkpoint file to an ASE readable file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"            , **argv,required=True , type=str     , help="input file")
    parser.add_argument("-if" , "--input_format"     , **argv,required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-fk" , "--forces_keyword"   , **argv,required=False, type=str     , help="keyword for the forces (default: %(default)s)" , default="forces")
    parser.add_argument("-o"  , "--output"           , **argv,required=True , type=str     , help="output file")
    parser.add_argument("-f"  , "--folder"           , **argv,required=False, type=str     , help="folder of the output files if each structure has to be saved in a different file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
def write_file(atoms: Atoms, ofile: str,forces_keyword: str = 'forces'):
    with open(ofile, "w") as f:
        # Write number of atoms
        f.write("  | Number of atoms                   :       {:d}\n".format(len(atoms)))
        
        # Write unit cell
        f.write("  Input geometry:\n")
        f.write("  | Unit cell:\n")
        cell = atoms.get_cell()
        for vector in cell:
            f.write("  |  {:18.8f} {:18.8f} {:18.8f}\n".format(*vector))
        
        # Write atomic positions
        f.write("  | Atomic structure:\n")
        f.write("  |       Atom                x [A]            y [A]            z [A]\n")
        for k, (symbol, position) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_positions()), start=1):
            f.write("  |    {:>5d}: Species {:<2s}    {:>18.8f} {:>18.8f} {:>18.8f}\n".format(k, symbol, *position))
        
        # Write forces if they exist
        if forces_keyword in atoms.arrays:
            f.write("\n  Total atomic forces [eV/Ang]:\n")
            for k, force in enumerate(atoms.arrays[forces_keyword], start=1):
                f.write("  |{:>5d}   {:>18.8f} {:>18.8f} {:>18.8f}\n".format(k, *force))

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures)
    print("\tn. of structures: {:d}".format(N))

    #------------------#
    if N > 1 and args.folder is None:
        raise ValueError("if the atomic structures are more than one, then you must provide the '-f,--folder' argument.")
    
    #------------------#
    print("\tWriting the atomic structures to file:")
    for n in range(N):
        
        atoms = structures[n]

        if args.folder is not None:       
            if not os.path.exists(args.folder):
                os.makedirs(args.folder)            
            ofile, file_extension  = os.path.splitext(args.output)
            ofile = f"{args.folder}/{ofile}.n={n}{file_extension}"
            ofile = os.path.normpath(ofile)
        else:
            ofile = args.output

        print("\t\t{:d}/{:d} --> {:s}".format(n+1,N,ofile))

        write_file(atoms,ofile,args.forces_keyword)


    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()