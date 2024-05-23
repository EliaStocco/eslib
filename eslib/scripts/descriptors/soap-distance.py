#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Compute a symmetric distance matrix based on SOAP descriptors."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-x"  , "--soap_descriptors", type=str     , required=True , **argv, help="file with the SOAP descriptors")
    parser.add_argument("-o"  , "--output"          , type=str     , required=True , **argv, help="output file with the selected structures")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print("\n\tReading SOAP descriptors from file '{:s}' ... ".format(args.soap_descriptors),end="")
    if str(args.soap_descriptors).endswith("npy"):
        X = np.load(args.soap_descriptors)
    elif str(args.soap_descriptors).endswith("txt"):
        X = np.loadtxt(args.soap_descriptors)
    print("done")

    #------------------#
    print("\n\tComputing the distance matrix from the SOAP descriptors ... ",end="")
    # this could be optimized
    D = X @ X.T
    assert D.shape[0] == X.shape[0]
    assert D.shape[1] == X.shape[0]
    print("done")

    #------------------#
    print("\tSaving distance matrix to file '{:s}' ... ".format(args.output),end="")
    if str(args.output).endswith("npy"):
        np.save(args.output, D)
    elif str(args.output).endswith("txt"):
        np.savetxt(args.output, D)
    print("done")   

#---------------------------------------#
if __name__ == "__main__":
    main()
