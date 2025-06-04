#!/usr/bin/env python
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Change the unit of a 'info' or 'array' in a trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--name"          , **argv, required=True , type=str, help="name for the new info/array")
    parser.add_argument("-w" , "--what"          , **argv, required=False, type=str, help="what the data is: 'i' (info) or 'a' (arrays) (default: %(default)s)", choices=["i","a"], default=None)
    parser.add_argument("-f" , "--family"        , **argv, required=True , type=str, help="family")
    parser.add_argument("-iu", "--in_unit"       , **argv, required=True , type=str, help="input unit")
    parser.add_argument("-ou", "--out_unit"      , **argv, required=True , type=str, help="output unit")
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file with the oxidation numbers (default: %(default)s)", default="oxidation-numbers.extxyz")
    parser.add_argument("-of", "--output_format" , **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
def correct(unit):
    if unit == "ang":
        return "angstrom"
    elif unit == "au":
        return "atomic_unit"
    else:
        return unit

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    args.in_unit  = correct(args.in_unit)
    args.out_unit = correct(args.out_unit)
    factor = convert(1,args.family,args.in_unit,args.out_unit)

    # print("\n\t{:>10s}: ".format("in-value"),args.value)
    print("\t{:>10s}: {:<s}".format("in-unit",args.in_unit))
    print("\t{:>10s}: {:<s}".format("out-unit",args.out_unit))
    print("\t{:>10s}: ".format("factor"),factor)
    # print("\t{:>10s}: ".format("out-value"),factor*args.value)

    # print("\n\t{:f} {:s} = {:f} {:s}".format(args.value,args.in_unit,factor*args.value,args.out_unit))

    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #------------------#
    if args.what is not None:
        if args.what == 'i':
            if args.name not in atoms[0].info:
                raise ValueError("{:s} not in 'info'".format(args.name))
        elif args.what == 'a':
            if args.name not in atoms[0].arrays:
                raise ValueError("{:s} not in 'arrays'".format(args.name))
        else:
            raise ValueError("coding error")
    else:
        pass
        #raise ValueError("coding error")


    #------------------#
    print("\tConverting '{:s}' from '{:s}' to '{:s}' ... ".format(args.name,args.in_unit,args.out_unit),end="")
    data = atoms.get(args.name)
    data *= factor
    atoms.set(args.name,data)
    # for structure in atoms:
    #     if args.what == 'i':
    #         structure.set(args.name,structure.info[args.name]*factor)
    #         structure.info[args.name] *= factor
    #     else:
    #         structure.arrays[args.name] *= factor
    print("done")
    
    #------------------#
    print("\n\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(file=args.output,format=args.output_format)
    

#---------------------------------------#
if __name__ == "__main__":
    main()