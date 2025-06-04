#!/usr/bin/env python
from ase.io import write

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import error, esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Rename an 'array' or 'info' in an extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="input file")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-on" , "--old_name"    , **argv, required=True , type=str, help="old name")
    parser.add_argument("-nn" , "--new_name"    , **argv, required=True , type=str, help="new name")    
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    print("\n\tOriginal information: ")
    df = atoms.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

    #---------------------------------------#
    print("\n\tLooking for '{:s}' in the trajectory fields ... ".format(args.old_name), end="")
    what = atoms.search(args.old_name)
    if what not in ['info','arrays']:
        print("{:s}: {:s} not found.".format(error,args.old_name))
        return -1
    print("done")

    print("\t'{:s}' was found in '{:s}'.".format(args.old_name,what))


    #---------------------------------------#
    # reshape
    print("\tChanging the name from '{:s}' to '{:s}'... ".format(args.old_name,args.new_name), end="")
    N = len(atoms)
    if what == "info":
        for n in range(N):
            atoms[n].info[args.new_name] = atoms[n].info.pop(args.old_name)
    elif what == "arrays":
        for n in range(N):
            atoms[n].arrays[args.new_name] = atoms[n].arrays.pop(args.old_name)
    else:
        raise ValueError("coding error")
    print("done")

    #---------------------------------------#
    print("\n\tFinal information: ")
    df = atoms.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

    #---------------------------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(file=args.output, format=args.output_format)
    print("done")
    
if __name__ == "__main__":
    main()

