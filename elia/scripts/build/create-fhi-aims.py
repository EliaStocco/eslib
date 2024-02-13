#!/usr/bin/env python
import argparse
import numpy as np
import os
import glob
from ase.io import read

# Description of the script's purpose
description = "Create the 'control.in' file for FHI-aims"

# Define the command-line argument parser with a description
parser = argparse.ArgumentParser(description=description)

parser.add_argument("-s" , "--species"      , action="store", type=str     , help="species"                             , default="light")
parser.add_argument("-f" , "--folder"       , action="store", type=str     , help="FHI-aims folder"                     , default=None)
parser.add_argument("-v" , "--variable"     , action="store", type=str     , help="bash variable for FHI-aims folder"   , default="AIMS_PATH")
parser.add_argument("-i" , "--input"        , action="store", type=str     , help="input file"                          , default="geometry.in")
parser.add_argument("-if", "--input_format" , action="store", type=str     , help="input file format (default: 'aims')" , default='aims')
parser.add_argument("-o" , "--output"       , action="store", type=str     , help="output file    "                     , default=None)

args = parser.parse_args()
end = ""

# Print the script's description
print("\n\t{:s}".format(description))

print("\tReading data from input file '{:s}' ... ".format(args.input), end=end)
atom = read(args.input,format=args.input_format)
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

species_folder = "{:s}/species_defaults/defaults_2020/{:s}".format(aims_folder,args.species)
print("\tReading chemical species species from '{:s}'".format(species_folder))

if args.output is None :
    args.output = "species.{:s}.in".format(args.species)

print("\tWriting output file '{:s}' ... ".format(args.output))
with open(args.output, "w") as target:
    for s in species:
        print("\t\tspecies '{:s}' ... ".format(s), end=end)

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

# Script completion message
print("\n\tJob done :)\n")
