#!/usr/bin/env python
import numpy as np
import os
import glob
from eslib.formatting import esfmt
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Create the 'species.in' file for FHI-aims."

#---------------------------------------#
def prepare_args(description):
    import argparse
    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str     , help="input file (default: %(default)s)"                          , default="geometry.in")
    parser.add_argument("-if", "--input_format" , **argv, type=str     , help="input file format (default: %(default)s)"                   , default=None)
    parser.add_argument("-b" , "--basis"        , **argv, type=str     , help="basis set (default: %(default)s)"                           , default="light")
    parser.add_argument("-f" , "--folder"       , **argv, type=str     , help="FHI-aims folder (default: %(default)s)"                     , default=None)
    parser.add_argument("-v" , "--variable"     , **argv, type=str     , help="bash variable for FHI-aims folder (default: %(default)s)"   , default="AIMS_PATH")
    parser.add_argument("-o" , "--output"       , **argv, type=str     , help="output file (default: %(default)s)"                         , default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):


    print("\tReading data from input file '{:s}' ... ".format(args.input), end="")
    atom = AtomicStructures.from_file(file=args.input, format=args.input_format,index=0)[0]
    print("done")

    species = np.unique(atom.get_chemical_symbols())
    print("\tExtracted chemical species: ",species)

    if args.folder is not None :
        aims_folder = args.folder
    else :
        try :
            aims_folder = os.environ.get(args.variable)
        except :
            raise ValueError("'FHI-aims' folder not found")

    if aims_folder is None :
        raise ValueError("'FHI-aims' folder not found (maybe you should 'source' some bash script ... )")
    print("\tFHI-aims folder: '{:s}'".format(aims_folder))

    species_folder = "{:s}/species_defaults/defaults_2020/{:s}".format(aims_folder,args.basis)
    print("\tReading chemical species from '{:s}'".format(species_folder))

    if args.output is None :
        args.output = "species.{:s}.in".format(args.basis)

    print("\tWriting output file '{:s}' ... ".format(args.output))
    with open(args.output, "w") as target:
        for s in species:
            print("\t\tspecies '{:s}' ... ".format(s), end="")

            pattern = "{:s}/*_{:s}_*".format(species_folder,s)
            files = glob.glob(pattern)
            if len(files) > 1 :
                raise ValueError("more than one file found for '{:s}'".format(s))
            elif len(files) == 0 :
                raise ValueError("no files found for '{:s}'".format(s))
            else :
                source_file = files[0]

            # Open each source file in read mode
            with open(source_file, "r") as source:
                # Read the content of the source file
                file_contents = source.read()
                
                # Write the content to the target file
                target.write(file_contents)

            print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()

