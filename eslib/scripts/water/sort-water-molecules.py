#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Template."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str  , help="molecule name (default: %(default)s)", default="molecule")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    molecules = structures.get(args.molecule)
    for n,(atoms,mol) in enumerate(zip(structures,molecules)):
        
        species = np.asarray([ 0 if a.symbol == "O" else 1 for a in atoms ])
        
        df = pd.DataFrame(columns=["index","molecule","species"])
        df["index"] = np.arange(len(species))
        df["molecule"] = mol
        df["species"] = species
        
        df = df.sort_values(by=["molecule","species"])
        ii = df["index"].to_numpy()
        structures[n] = atoms[ii]
    
    print("\n\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

    

#---------------------------------------#
if __name__ == "__main__":
    main()