#!/usr/bin/env python
import numpy as np
from eslib.tools import string2function
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Apply a function to an array read from a txt file, and save the result to another txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , required=True,**argv,type=str, help="txt input file")
    parser.add_argument("-f" , "--function"     , required=True,**argv,type=str, help="source code of the function to be applied")
    parser.add_argument("-o" , "--output"       , required=True,**argv,type=str, help="txt output file")
    parser.add_argument("-of", "--output_format", required=False,**argv,type=str, help="txt output format for np.savetxt (default: %(default)s)", default='%24.18f')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    print("\tConverting string into function ... ", end="")
    function = string2function(args.function)
    print("done")

    #---------------------------------------#
    print("\tReading array from file '{:s}' ... ".format(args.input), end="")
    inarray = np.loadtxt(args.input)
    print("done")

    print("\tinput array shape: ",inarray.shape)

    #---------------------------------------#
    print("\tApplying function to the array ... ", end="")
    outarray = function(inarray)
    print("done")

    print("\toutput array shape: ",outarray.shape)
    
    #---------------------------------------#
    if args.output is None:
        print("\t{:s}: no output file provided.\nSpecify it with -o,--output")
    else:
        print("\tSave output array to file '{:s}' ... ".format(args.output), end="")
        np.savetxt(args.output,outarray,fmt=args.output_format)
        print("done")

if __name__ == "__main__":
    main()

