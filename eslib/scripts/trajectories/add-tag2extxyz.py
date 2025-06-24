#!/usr/bin/env python
import json, os
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import warning, esfmt
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool
from eslib.show import show_dict


#---------------------------------------#
# Description of the script's purpose
description = "Add tags (info) to an extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-t" , "--tags"         , **argv, type=str, required=True , help="JSON formatted tags or JSON file")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)

    #---------------------------------------#
    # data
    print("\tReading tags ... ", end="")
    if str(args.tags).endswith(".json") and os.path.isfile(args.tags):
        with open(args.tags, "r") as f:
            tags:dict = json.load(f)
    else:
        tags:dict = json.loads(args.tags)
    print("done")
    print("\tTags:")
    show_dict(tags,string="\t",width=10)
    
    #---------------------------------------#
    print("\tStoring tags ... ", end="")
    for k,v in tags.items():
        name = f"tag:{k}"
        structures.set(name,np.full(N,v),"info")
    print("done")
    
    #---------------------------------------#
    print("\n\tWriting trajectory to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")

if __name__ == "__main__":
    main()

