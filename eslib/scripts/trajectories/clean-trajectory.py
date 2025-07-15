#!/usr/bin/env python
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import warning, esfmt, error
from eslib.functions import get_file_size_human_readable
from eslib.input import slist

#---------------------------------------#
# Description of the script's purpose
description = "Clean a trajectory by removing all info and arrays."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str  , required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str  , required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keep"         , **argv, type=slist, required=False, help="info/arrays to be kept (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"       , **argv, type=str  , required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str  , required=False,  help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    try:
        value, unit = get_file_size_human_readable(args.input)
        print(f"\n\tInput file size: {value} {unit}")
    except:
        print(f"\n\t{warning}: an error occurred while retrieving the input file size")

    #------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input),end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)
    
    #------------------#
    # summary
    print("\n\t  Summary of the properties: ")
    try:
        df = structures.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t "))
    except:
        print(f"\t{error}: an error occurred while retrieving the properties")

    #------------------#
    # data
    if args.keep is None:
        args.keep = []
    print("\n\tRemoving info:")
    for key in structures.get_keys("info"):
        if key in args.keep:
            print("\t - keeping '{:s}'".format(key))
        else:
            print("\t - removing '{:s}'".format(key))
            structures.remove_info_array(key,what="info")
            
    print("\n\tRemoving arrays:")
    for key in structures.get_keys("arrays"):
        if key in ["positions","numbers",*args.keep]:
            print("\t - keeping '{:s}'".format(key))
        else:
            print("\t - removing '{:s}'".format(key))
            structures.remove_info_array(key,what="arrays")
            
    #------------------#
    # summary
    print("\n\t  Summary of the properties: ")
    try:
        df = structures.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t "))
    except:
        print(f"\t{error}: an error occurred while retrieving the properties")
    
    #------------------#
    # output
    print("\n\tWriting trajectory to file '{:s}' ... ".format(args.output),end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
    
    #------------------#
    try:
        value, unit = get_file_size_human_readable(args.output)
        print(f"\tOutput file size: {value} {unit}")
    except:
        print(f"\t{warning}: an error occurred while retrieving the input file size")

if __name__ == "__main__":
    main()

