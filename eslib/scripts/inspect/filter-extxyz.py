#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import string2function

#---------------------------------------#
# Description of the script's purpose
description = "Filter an extxyz."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-f" , "--function"     , required=True,**argv,type=str, help="source code of the function to be applied")
    parser.add_argument("-k" , "--keyword"     , **argv, required=True, type=str, help="keyword")
    parser.add_argument("-o" , "--output"       , required=True,**argv,type=str, help="txt output file")
    parser.add_argument("-of", "--output_format", required=False,**argv,type=str, help="txt output format for np.savetxt (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tConverting string into function ... ", end="")
    function = string2function(args.function)
    print("done")

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structure = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tNumber of structures: ", len(structure))

    #------------------#
    print("\tExtracting '{:s}' from the structure ... ".format(args.keyword), end="")
    data = structure.get(args.keyword)
    print("done")
    print("\tdata shape: ", data.shape)

    #------------------#
    print("\tApplying function to the info/array ... ", end="")
    outarray = function(data)
    print("done")
    assert isinstance(outarray, np.ndarray), "The function must return a numpy array"
    assert outarray.ndim == 1, "The function must return a 1D numpy array"
    assert outarray.dtype == bool, "The function must return a boolean array"

    indices = np.where(outarray)[0]
    print("\tNumber of True values: ", len(indices))
    
    #------------------#
    print("\tSubsampling the structures ... ", end="")
    structure = structure.subsample(indices)
    print("done")
    print("\tNumber of structures: ", len(structure))
    
    #------------------#
    print("\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    structure.to_file(file=args.output, format=args.output_format)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()

