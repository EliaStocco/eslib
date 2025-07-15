#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functional import extend2NDarray

AtomicStructures.set_parallel(False)

#---------------------------------------#
# Description of the script's purpose
description = "Compute the time derivative of a info/array."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, type=str  , required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format"  , **argv, type=str  , required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"       , **argv, type=str  , required=True , help="keyword of the info/array")
    parser.add_argument("-dt", "--time_step"     , **argv, type=float, required=True , help="time step [fs]")
    parser.add_argument("-ok", "--output_keyword", **argv, type=str  , required=True , help="keyword of the time derivative")
    parser.add_argument("-o" , "--output"        , **argv, type=str  , required=True , help="output file")
    parser.add_argument("-of", "--output_format" , **argv, type=str  , required=False, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)
    
    assert structures.is_trajectory(), "Input file must be a trajectory."
    
    #------------------#
    # check if the keyword exists
    print("\n\tExtracting data for keyword '{:s}' ... ".format(args.keyword), end="")
    # if not structures.has(args.keyword):
    #     raise ValueError(f"Keyword '{args.keyword}' not found in the input file.")
    # else:
    what = structures.search(args.keyword)
    assert what == "info" or what == "arrays", \
        f"Keyword '{args.keyword}' must be in 'info' or 'arrays', not '{what}'."
    
    data = structures.get(args.keyword,what=what)
    # if not isinstance(data, np.ndarray):
    #     raise TypeError(f"Data for keyword '{args.keyword}' must be a numpy array, not {type(data)}.")
    print("done")
    print("\tdata.shape: ", data.shape)
    
    #------------------#
    # compute the time derivative
    print("\n\tComputing time derivative ... ", end="")
    if data.ndim == 1:
        data = np.gradient(data, axis=0)/args.time_step
    elif data.ndim in [2,3]:
        gradient = extend2NDarray(lambda x: np.gradient(x)/args.time_step)
        data = gradient(data,axis=0)
    else:
        raise ValueError(f"Data for keyword '{args.keyword}' must be 1D, 2D, or 3D, not {data.ndim}D.")
    
    print("done")
    print("\tgrad.shape: ", data.shape)
    
    #------------------#
    # store the time derivative
    print("\n\tStoring time derivative in keyword '{:s}' ... ".format(args.output_keyword), end="")
    structures.set(args.output_keyword, data,what)
    print("done")
    
    #------------------#
    # save the output file
    print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")
    
    return

#---------------------------------------#
if __name__ == "__main__":
    main()

