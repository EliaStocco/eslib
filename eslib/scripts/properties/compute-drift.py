#!/usr/bin/env python
import numpy as np

from eslib.classes.properties import Properties
from eslib.formatting import esfmt
from eslib.input import flist
from eslib.tools import convert

#---------------------------------------#
description = "Compute the drift."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # Input
    parser.add_argument("-i" , "--input"          , **argv, type=str  , required=True , help="input file")
    # Keywords
    parser.add_argument("-c" , "--conserved"      , **argv, type=str  , required=False, help="`conserved` keyword (default: %(default)s)", default="conserved")
    # Units
    parser.add_argument("-u" , "--unit", **argv, type=str  , required=False, help="`conserved` unit (default: %(default)s)", default="atomic_unit")
    # Time step
    parser.add_argument("-dt" , "--time_step"     , **argv, type=float  , required=False, help="time step [fs](default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    # atomic structures
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    properties = Properties.from_file(file=args.input)
    print("done\n")
    print("\tn. of snapshots: {:d}".format(len(properties)))

    #------------------#
    # summary
    print("\n\tSummary of the properties: ")
    df = properties.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    print()

    #------------------#
    # Extraction/Construction
    print("\tExtracting 'conserved' using keyword '{:s}' ... ".format(args.conserved), end="")
    conserved = properties.get(args.conserved)
    print("done")
    print("\t'conserved'.shape: ",conserved.shape)

    #------------------#
    # Conversion
    if args.unit not in ["","au","atomic_unit"]:
        print("\n\tConverting 'conserved' to 'atomic_unit' ... ", end="")
        conserved = convert(conserved,"energy",args.unit,"atomic_unit")
        print("done")

    if args.time_step is None:
        print("\n\tEstimating time step: ",end="")
        time = properties.get("time","femtosecond")
        args.time_step = np.mean(np.diff(time))
        print("{:.2f} fs".format(args.time_step))


    #------------------#
    # Drift
    # conserving `conserved` and `Econserved` to meV
    conserved = convert(conserved,"energy","atomic_unit","millielectronvolt")

    print("\n\tFitting 'conserved' with a line ... ", end="")
    x = np.arange(len(conserved))*args.time_step
    x = convert(x,"time","femtosecond","picosecond")
    slope, _ = np.polyfit(x, conserved, 1)
    print("done")
    print(f"\t slope: {slope:>.6f} meV/ps")

    std = np.std(conserved)
    print(f"\t   std: {std:>.6f} meV")


#---------------------------------------#
if __name__ == "__main__":
    main()
