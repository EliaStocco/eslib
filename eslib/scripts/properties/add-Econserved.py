#!/usr/bin/env python
import numpy as np
from eslib.classes.properties import Properties
from eslib.formatting import esfmt
from eslib.input import flist
from eslib.tools import convert

#---------------------------------------#
description = "Compute the actual conserved quantity of the system when an electric field is applied."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # Input
    parser.add_argument("-i" , "--input"          , **argv, type=str  , required=True , help="input file")
    # Keywords
    parser.add_argument("-c" , "--conserved"      , **argv, type=str  , required=False, help="`conserved` keyword (default: %(default)s)", default="conserved")
    parser.add_argument("-d" , "--dipole"         , **argv, type=str  , required=False, help="`dipole` keyword (default: %(default)s)", default="dipole")
    parser.add_argument("-e" , "--efield"         , **argv, type=str  , required=False, help="`Efield` keyword (default: %(default)s)", default="Efield")
    parser.add_argument("-ec", "--efield_constant", **argv, type=flist, required=False, help="Efield value when constant(default: %(default)s)", default=None)
    # Units
    parser.add_argument("-cu" , "--conserved_unit", **argv, type=str  , required=False, help="`conserved` unit (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-du" , "--dipole_unit"   , **argv, type=str  , required=False, help="`dipole` unit (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-eu" , "--efield_unit"   , **argv, type=str  , required=False, help="`Efield` unit (default: %(default)s)", default="atomic_unit")
    # Time step
    parser.add_argument("-dt" , "--time_step"     , **argv, type=float  , required=False, help="time step [fs](default: %(default)s)", default=1)
    # Output
    parser.add_argument("-on", "--output_name"    , **argv, type=str  , required=False, help="output `Econserved` keyword (default: %(default)s)", default="Econserved")
    parser.add_argument("-o" , "--output"         , **argv, type=str  , required=True , help="output file")
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

    print("\tExtracting 'dipole' using keyword '{:s}' ... ".format(args.dipole), end="")
    dipole = properties.get(args.dipole)
    print("done")

    if args.efield_constant is None:
        print("\tExtracting 'Efield' using keyword '{:s}' ... ".format(args.efield), end="")
        efield = properties.get(args.efield)
        print("done")
    else:
        print("\tConstructing 'Efield' ... ", end="")
        efield = np.asarray(args.efield_constant)
        assert efield.shape == (3,), "efield shape must be (3,)"
        efield = np.tile(efield, (len(properties), 1))
        print("done")

    print()
    print("\t'conserved'.shape: ",conserved.shape)
    print("\t   'dipole'.shape: ",dipole.shape)
    print("\t   'Efield'.shape: ",efield.shape)

    assert dipole.shape == efield.shape, "dipole and efield must have the same shape"
    assert len(conserved) == dipole.shape[0], "conserved and dipole must have the same length"
    assert conserved.ndim == 1, "conserved must be a 1D array"

    #------------------#
    # Conversion
    print()
    if args.conserved_unit not in ["","au","atomic_unit"]:
        print("\tConverting 'conserved' to 'atomic_unit' ... ", end="")
        conserved = convert(conserved,"energy",args.conserved_unit,"atomic_unit")
        print("done")

    if args.dipole_unit not in ["","au","atomic_unit"]:
        print("\tConverting 'dipole' to 'atomic_unit' ... ", end="")
        dipole = convert(dipole,"electric-dipole",args.dipole_unit,"atomic_unit")
        print("done")

    if args.efield_unit not in ["","au","atomic_unit"]:
        print("\tConverting 'efield' to 'atomic_unit' ... ", end="")
        efield = convert(efield,"electric-field",args.efield_unit,"atomic_unit")
        print("done")

    #------------------#
    # Computation
    print("\tComputing 'Econserved' ... ", end="")
    eda = (dipole * efield).sum(axis=1)
    Econserved = conserved - eda
    print("done")

    #------------------#
    # Drift
    # conserving `conserved` and `Econserved` to meV
    conserved = convert(conserved,"energy","atomic_unit","millielectronvolt")
    Econserved = convert(Econserved,"energy","atomic_unit","millielectronvolt")

    print("\n\tFitting 'conserved and 'Econserved' with a line ... ", end="")
    x = np.arange(len(Econserved))*args.time_step
    x = convert(x,"time","femtosecond","picosecond")
    slope, _ = np.polyfit(x, conserved, 1)
    Eslope, _ = np.polyfit(x, Econserved, 1)
    print("done")
    print(f"\t conserved slope: {slope:.6f} meV/ps")
    print(f"\tEconserved slope: {Eslope:.6f} meV/ps")



#---------------------------------------#
if __name__ == "__main__":
    main()
