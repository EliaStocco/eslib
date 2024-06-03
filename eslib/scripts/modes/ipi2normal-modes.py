#!/usr/bin/env python
from eslib.classes.normal_modes import NormalModes
from ase.io import read
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Prepare the necessary file to project a MD trajectory onto phonon modes: read results from i-PI."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-r",  "--reference",       type=str, **argv, help="reference structure w.r.t. which the phonons are computed [a.u.] (default: %(default)s)",default=None)
    parser.add_argument("-f",  "--folder",         type=str, **argv, help="folder with the output files of the i-PI normal analysis (default: %(default)s)", default="vib")
    parser.add_argument("-o",  "--output",        type=str, **argv, help="output file (default: %(default)s)", default="normal-modes.pickle")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # read reference atomic structure
    reference = None
    if args.reference is not None:
        print("\tReading reference atomic structure from input '{:s}' ... ".format(args.reference), end="")
        reference = read(args.reference,index=0)
        print("done")

    #---------------------------------------#
    # phonon modes
    print("\n\tReading normal modes from folder '{:s}' ... ".format(args.folder),end="")
    nm = NormalModes.from_folder(folder=args.folder,ref=reference)
    print("done")

    #---------------------------------------#
    print("\n\tWriting normal modes to file '{:s}' ... ".format(args.output), end="")
    nm.to_pickle(args.output)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
