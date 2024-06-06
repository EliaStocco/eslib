#!/usr/bin/env python
from eslib.tools import convert
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Convert a physical quantity from one measure unit to another."

#---------------------------------------#
def correct(unit):
    if unit == "ang":
        return "angstrom"
    elif unit == "au":
        return "atomic_unit"
    else:
        return unit
    
#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-v", "--value", type=float, required=True, **argv, help="value")
    parser.add_argument("-f", "--family", type=str, required=False,**argv, help="family (default: %(default)s)", default="length")
    parser.add_argument("-iu", "--in_unit", type=str, required=False,**argv, help="input unit (default: %(default)s)", default="au")
    parser.add_argument("-ou", "--out_unit", type=str, required=False,**argv, help="output unit (default: %(default)s)", default="ang")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    if args.family is None:
        raise ValueError("'family' can not be None")
    
    args.in_unit  = correct(args.in_unit)
    args.out_unit = correct(args.out_unit)
    factor = convert(1,args.family,args.in_unit,args.out_unit)

    print("\n\t{:>10s}: ".format("in-value"),args.value)
    print("\t{:>10s}: {:<s}".format("in-unit",args.in_unit))
    print("\t{:>10s}: {:<s}".format("out-unit",args.out_unit))
    print("\t{:>10s}: ".format("factor"),factor)
    print("\t{:>10s}: ".format("out-value"),factor*args.value)

    print("\n\t{:f} {:s} = {:f} {:s}".format(args.value,args.in_unit,factor*args.value,args.out_unit))

#---------------------------------------#
if __name__ == "__main__":
    main()