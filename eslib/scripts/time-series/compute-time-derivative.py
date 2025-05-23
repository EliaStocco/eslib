#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import warning, esfmt, eslog
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Compute the time derivative of a info/array."
documentation = "The user has to provide the time step in fs.\n\
The time derivative will have the unit of [in_name]/fs,\n\
where [in_name] is the unit of the info/array whose derivative will be computed."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str  , required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str  , required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-dt", "--time_step"    , **argv, type=float, required=True , help="time step [fs]")
    parser.add_argument("-in", "--in_name"      , **argv, type=str  , required=True , help="name of the info/array")
    parser.add_argument("-on", "--out_name"     , **argv, type=str  , required=True , help="name of the time derivative of the info/array")
    parser.add_argument("-o" , "--output"       , **argv, type=str  , required=True , help="output file (default: %(default)s)", default="output.extxyz")
    parser.add_argument("-of", "--output_format", **argv, type=str  , required=False, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):

    #-------------------#
    # atomic structures
    with eslog(f"Reading atomic structures from file '{args.input}'"):
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    N = len(structures) 
    print("\tn. of atomic structures: ",N)

    #-------------------#
    with eslog(f"Extracting '{args.in_name}'"):
        what = structures.search(args.in_name) # info or array
        data = structures.get(args.in_name,what=what)
    print(f"\t data.shape: {data.shape}")
    
    #-------------------#
    with eslog("\nComputing time derivative"):
        data = np.gradient(data,axis=0)/args.time_step
    print(f"\t data.shape: {data.shape}")   
    
    #-------------------#
    with eslog(f"\nSaving the time derivative in '{args.out_name}'"):
        structures.set(args.out_name,data,what)
    
    #-------------------#
    with eslog(f"\nSaving structures to file '{args.output}'"):
        structures.to_file(file=args.output,format=args.output_format)
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()

