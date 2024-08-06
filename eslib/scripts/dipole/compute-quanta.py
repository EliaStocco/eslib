#!/usr/bin/env python
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, error
from eslib.physics import compute_dipole_quanta

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dipole quanta."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"    , **argv, type=str, help="extxyz file with the unfolded atomic configurations [a.u]")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-id", "--in_dipole"  , **argv, type=str, help="name of the input dipoles (default: %(default)s)", default='dipole')
    parser.add_argument("-oq", "--out_quanta"  , **argv, type=str, help="name of the output quanta (default: %(default)s)", default='quanta')
    parser.add_argument("-o" , "--output"   , **argv, type=str, help="output file with the fixed trajectory (default: %(default)s)", default="trajectory.fixed.extxyz")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(structures)))

    #------------------#
    if args.in_dipole == args.out_quanta:
        print("\t{:s}: specify -id,--in_dipole ('{:s}') different from -oq,--out_quanta ('{:s}').".format(error,args.in_dipole,args.out_quanta))
        return -1

    #------------------#
    print("\n\tComputing the dipole quanta  ... ", end="")
    structures,_ = compute_dipole_quanta(structures,in_keyword=args.in_dipole,out_keyword=args.out_quanta)
    print("done")

    #------------------#
    print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")    

#---------------------------------------#
if __name__ == "__main__":
    main()