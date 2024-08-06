#!/usr/bin/env python
import numpy as np
from eslib.classes.properties import Properties
from eslib.formatting import esfmt

#---------------------------------------#
description = "Summary of an MD trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input", **argv, type=str, help="input file")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    atoms = Properties.from_file(file=args.input)
    print("done\n")
    print("\tn. of snapshots: {:d}".format(len(atoms)))

    #---------------------------------------#
    # summary
    print("\n\tSummary of the properties: ")
    df = atoms.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

#---------------------------------------#
if __name__ == "__main__":
    main()
