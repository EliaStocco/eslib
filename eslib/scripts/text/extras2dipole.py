#!/usr/bin/env python
import json
from eslib.formatting import esfmt, float_format
import numpy as np

#---------------------------------------#
# Description of the script's purpose
description = "Read the dipoles from a i-PI extras file and save them in a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv, required=True , type=str, help="input file with the JSOn formatted dipoles [au]")
    parser.add_argument("-o" , "--output", **argv, required=False, type=str, help="txt output file with dipoles (default: 'dipoles.txt')", default='dipoles.txt')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading dipoles from file '{:s}' ... ".format(args.input), end="")
    # Open the input file for reading
    with open(args.input, 'r') as f:
        lines = f.readlines()

    # Initialize an empty list to store dipole values
    dipoles = []

    # Iterate through each line in the file
    for line in lines:
        # Check if the line contains dipole data
        if '"dipole"' in line:
            # Extract the dipole values from the line
            dipole_data = json.loads(line.strip().split('dipole": ')[1][:-1])
            # Append the dipole values to the list
            dipoles.append(dipole_data)
    print("done")

    #------------------#
    print("\tConverting dipoles to np.array: ... ", end="")
    dipoles = np.asarray(dipoles).reshape((-1,3))
    print("done")
    print("\tdipoles shape: ",dipoles.shape, end="")

    #------------------#
    print("\n\tWriting dipoles to file '{:s}' ... ".format(args.output), end="")
    try:
        np.savetxt(args.output,dipoles,fmt=float_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))


#---------------------------------------#
if __name__ == "__main__":
    main()