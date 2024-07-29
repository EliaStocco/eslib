#!/usr/bin/env python
import json
from eslib.formatting import esfmt, float_format, warning
from eslib.input import str2bool
import numpy as np
import re
from eslib.classes.physical_tensor import PhysicalTensor

#---------------------------------------#
# Description of the script's purpose
description = "Read the dipoles from a i-PI extras file and save them in a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv, required=True , type=str, help="input file with the JSOn formatted dipoles [au]")
    parser.add_argument("-k" , "--keyword" , **argv, required=False , type=str, help="keyword (default: %(default)s)", default='dipole')
    parser.add_argument("-rr" , "--remove_replicas", **argv,required=False, type=str2bool, help='whether to remove replicas (default: false)', default=False)
    parser.add_argument("-o" , "--output", **argv, required=False, type=str, help="txt output file with dipoles (default: %(default)s)", default='dipoles.txt')
    return parser# .parse_args()

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
    steps = []
    step = re.compile(r"Step:\s+(\d+)")
    # strings = [ step.search(comment).group(1) for comment in comments ]
    #         steps = np.asarray([int(i) for i in strings],dtype=int)

    # Iterate through each line in the file
    for line in lines:
        test = step.search(line)
        if test is not None:
            steps.append(int(test.group(1)))
        # Check if the line contains dipole data
        if '\"{:s}\"'.format(args.keyword) in line:
            # Extract the dipole values from the line
            json_data = json.loads(line.replace("\n","").replace('\x00',""))
            dipole_data = json_data[args.keyword] #json.loads(line.strip().split('"dipole":')[1][:-1])
            # Append the dipole values to the list
            dipoles.append(dipole_data)
    print("done")

    #------------------#
    steps= np.asarray(steps)
    test, indices = np.unique(steps, return_index=True)
    if steps.shape != test.shape and not args.remove_replicas:
        print("\t{:s}: there could be replicas. Specify '-rr/--remove_replicas true' to remove them.".format(warning))
    if args.remove_replicas:
        dipoles = [dipoles[index] for index in indices]

    #------------------#
    print("\tConverting dipoles to np.array: ... ", end="")
    dipoles = np.asarray(dipoles).reshape((-1,3))
    print("done")
    print("\tdipoles shape: ",dipoles.shape, end="")

    #------------------#
    # print("\n\tWriting dipoles to file '{:s}' ... ".format(args.output), end="")
    # try:
    #     np.savetxt(args.output,dipoles,fmt=float_format)
    #     print("done")
    # except Exception as e:
    #     print("\n\tError: {:s}".format(e))

    #---------------------------------------#
    print("\n\tConverting data into PhysicalTensor ... ", end="")
    dipoles = PhysicalTensor(dipoles)
    print("done")

    print("\n\tWriting dipoles to file '{:s}' ... ".format(args.output), end="")
    dipoles.to_file(file=args.output,fmt=float_format)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()