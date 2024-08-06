#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import numpy as np
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist
from ase.spacegroup import get_spacegroup

#---------------------------------------#
description = "Find the symmetry of an atomic structure."
    
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"       , type=str  , **argv, required=True , help="atomic structure input file")
    parser.add_argument("-if", "--input_format", type=str  , **argv, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-t" , "--threshold"   , type=flist, **argv, required=False, help="list of thresholds (default: %(default)s)" , default=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    args.threshold = np.asarray(args.threshold)
    args.threshold.sort()

    #------------------#
    print()
    line = "|{:^15s}|{:^15s}|{:^15s}|{:^15s}|".format("Threshold","Spacegroup","Spacegroup","n. of sym.")
    N = len(line)-2
    print("\t|"+"-"*N+"|")
    print("\t"+line)
    line = "|{:^15s}|{:^15s}|{:^15s}|{:^15s}|".format("","symbol","number","operations")
    print("\t"+line)
    print("\t|"+"-"*N+"|")
    for symprec in args.threshold:
        spacegroup = get_spacegroup(atoms,symprec=symprec)
        line = "|{:>12.2e}   |{:^15s}|{:^15d}|{:^15d}|".format(symprec,spacegroup.symbol,spacegroup.no,spacegroup.nsymop)
        print("\t"+line)
        # print("\tThreshold: {:>.2e}  Spacegroup: {:>6s}".format(symprec,spacegroup.symbol,spacegroup.no,spacegroup.nsymop))
    print("\t|"+"-"*N+"|")
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()