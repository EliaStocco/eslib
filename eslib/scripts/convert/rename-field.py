#!/usr/bin/env python
from ase.io import write
from eslib.classes.trajectory import trajectory, info, array
from eslib.formatting import esfmt, error

#---------------------------------------#
# Description of the script's purpose
description = "Save an 'array' or 'info' from an extxyz file to a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="input file [extxyz]")
    parser.add_argument("-n" , "--name"         , **argv, required=True , type=str, help="name for the new info/array")
    parser.add_argument("-r" , "--renamed"      , **argv, required=True , type=str, help="new name for the new info/array")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format", default=None)
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = trajectory(args.input)
    print("done")

    #---------------------------------------#
    print("\tLooking for '{:s}' in the trajectory fields: ".format(args.name), end="")
    if args.name in atoms[0].info:
        print("'{:s}' is in 'info'".format(args.name))
        what = "info"
    elif args.name in atoms[0].arrays:
        print("'{:s}' is in 'arrays'".format(args.name))
        what = "array"
    else:
        print("{:s}: {:s} not found.".format(error,args.name))
        return

    #---------------------------------------#
    # reshape
    print("\tChanging the name of '{:s}' to '{:s}'... ".format(args.name,args.renamed), end="")
    N = len(atoms)
    if what == "info":
        for n in range(N):
            atoms[n].info[args.renamed] = atoms[n].info.pop(args.name)
    elif what == "array":
        for n in range(N):
            atoms[n].arrays[args.renamed] = atoms[n].arrays.pop(args.name)
    print("done")

    #---------------------------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=list(atoms),filename=args.output, format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))
if __name__ == "__main__":
    main()

