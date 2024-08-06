#!/usr/bin/env python
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, error
from eslib.input import slist

#---------------------------------------#
# Description of the script's purpose
description = "Remove 'array' or 'info' from an extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="input file")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--name"         , **argv, required=True , type=slist, help="name for the new info/array")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    print("\n\tOriginal information: ")
    df = trajectory.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

    #---------------------------------------#
    for name in args.name:
        print("\n\tLooking for '{:s}' in the trajectory fields ... ".format(name), end="")
        what = trajectory.search(name)
        if what not in ['info','arrays']:
            print("{:s}: {:s} not found.".format(error,name))
            return -1
        print("done")

        print("\t'{:s}' was found in '{:s}'.".format(name,what))

        #---------------------------------------#
        print("\tDeleting '{:s}' from the trajectory ... ".format(name), end="")
        for atoms in trajectory:
            if what == 'info':
                del atoms.info[name]
            else:
                del atoms.arrays[name]
        print("done")

    #---------------------------------------#
    print("\n\tFinal information: ")
    df = trajectory.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    
    #---------------------------------------#
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    trajectory.to_file(file=args.output, format=args.output_format)
    print("done")

if __name__ == "__main__":
    main()

