#!/usr/bin/env python
import re

import numpy as np

from eslib.formatting import esfmt, warning
from eslib.input import str2bool
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Extract the values of the dipole from a file written by FHI-aims and convert to atomic_unit."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"         , **argv,type=str, help="input txt file")
    parser.add_argument("-rr" , "--remove_replicas", **argv,required=False, type=str2bool, help='whether to remove replicas (default: false)', default=False)
    parser.add_argument("-o", "--output"        , **argv,type=str, help="output file with the dipole values (default: %(default)s)", default="dipole.aims.txt")
    parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: %(default)s)", default='%24.18f')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # Open the input file for reading and the output file for writing
    factor = convert(1,"length","angstrom","atomic_unit")
    n = 0 
    dipoles = []
    steps = []
    step = re.compile(r"Step:\s+(\d+)")
    pattern = re.compile(r'Total dipole moment.*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?).*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?).*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)')
    with open(args.input, 'r') as input_file:
        # Iterate through each line in the input file
        for line in input_file:
            test = step.search(line)
            if test is not None:
                steps.append(int(test.group(1)))

            if "Total dipole moment" in line and "[eAng]" in line:
                # Search for the pattern and extract the first three float values
                # matches = re.search(r"Total dipole moment \[eAng\]\s*:\s*([-+]?\d*\.\d+|\d+\.\d*|\d+)", line)
                
                # If the pattern is found, extract and write the values to the output file
                # line = line.replace("E","e")
                matches = re.search(pattern, line)
        
                # If the pattern is found, extract and return the first three float values
                if matches:
                    float_values = [float(match) for match in matches.groups()[:3]]
                    float_values = np.asarray(float_values).reshape((1,3))
                    dipoles.append(float_values)
                    # np.savetxt(output_file,factor*float_values,fmt=args.output_format)
                    # n += 1 
                    # output_file.write(','.join(map(str, float_values)) + '\n')  # Save the values as a comma-separated line
    print("\tN. of dipole values found: ",len(dipoles))

    #------------------#
    steps= np.asarray(steps)
    test, indices = np.unique(steps, return_index=True)
    if steps.shape != test.shape and not args.remove_replicas:
        print("\t{:s}: there could be replicas. Specify '-rr/--remove_replicas true' to remove them.".format(warning))
    if args.remove_replicas:
        dipoles = [dipoles[index] for index in indices]

    print("\n\tWriting dipoles to file '{:s}' ... ".format(args.output),end="")
    with open(args.output, 'w') as output_file:
        dipoles = np.asarray(dipoles).reshape(-1,3)
        np.savetxt(output_file,factor*dipoles,fmt=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()