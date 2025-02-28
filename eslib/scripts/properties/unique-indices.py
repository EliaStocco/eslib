#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt, everythingok, warning

#---------------------------------------#
# Description of the script's purpose
description = "Take only the unique indices."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , type=str, **argv, required=True, help='txt input file')
    parser.add_argument("-o", "--output", type=str, **argv, required=True, help='txt output file')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading indices from file '{:s}' ... ".format(args.input), end="")
    indices = np.loadtxt(args.input).astype(int)
    print("done")
    assert indices.ndim == 1, "The input file should contain only one column."
    print("\n\tNumber of indices : {:d}".format(len(indices)))

    #------------------#
    unique,ii = np.unique(indices, return_index=True)
    assert np.allclose(unique,indices[ii]), "The indices are not unique."
    print("\tNumber of unique indices : {:d}".format(len(unique)))
    
    #------------------#
    test = np.arange(0,len(ii),dtype=int)
    ok = np.allclose(unique,test)
    if not ok:
        print("\n\t{:s}: the indices are not consecutive.".format(warning))
    else:
        print("\n\t{:s}: the indices are consecutive.".format(everythingok))    
    
    #------------------#
    print("\n\tWriting unique indices to file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output, ii, fmt="%d")
    print("done")
    
    pass

#---------------------------------------#
if __name__ == "__main__":
    main()
