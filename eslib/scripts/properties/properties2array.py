#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
from eslib.classes.properties import Properties
from eslib.functions import suppress_output
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Extract a property from a pickle to a txt/npy file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , type=str     , **argv, required=True , help='pickle file with the properties')
    parser.add_argument("-n" , "--name"          , **argv, required=True , type=str, help="name of the property to extract")
    parser.add_argument("-f" , "--family"        , **argv, required=False , type=str, help="family (default: %(default)s)", default=None)
    parser.add_argument("-iu", "--in_unit"       , **argv, required=False , type=str, help="input unit (default: %(default)s)", default=None)
    parser.add_argument("-ou", "--out_unit"      , **argv, required=False , type=str, help="output unit (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"        , type=str     , **argv, required=True , help='npy/txt output file')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    with suppress_output():
        properties = Properties.from_pickle(file_path=args.input)
    print("done")

    #------------------#
    print("\tExtracting property '{:s}' ... ".format(args.name), end="")
    data = properties[args.name]
    print("done")
    print("\t'{:s}' shape: :".format(args.name),data.shape)

    #------------------#
    if np.all( [a is not None for a in [args.family,args.in_unit,args.out_unit] ]): 
        factor = convert(1,args.family,args.in_unit,args.out_unit)
        print("\t{:>10s}: {:<s}".format("in-unit",args.in_unit))
        print("\t{:>10s}: {:<s}".format("out-unit",args.out_unit))
        print("\t{:>10s}: ".format("factor"),factor)
        data *= factor
    elif np.any( [a is not None for a in [args.family,args.in_unit,args.out_unit] ]): 
        raise ValueError("specify all the options")

    #------------------#
    print("\tSaving property to file '{:s}' ... ".format(args.output), end="")
    args.output = str(args.output)
    if args.output.endswith("npy"):
        np.save(args.output,data)
    else:
        np.savetxt(args.output,data)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
