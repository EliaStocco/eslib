#!/usr/bin/env python
import re
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, eslog, message, warning
from eslib.input import ilist

#---------------------------------------#
# Description of the script's purpose
description = "Split an extxyz file into two files depending if its info/arrays contain or not numpy.nan values."

# TODO: still to debug

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str, help="input file [extxyz]")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o"  , "--output"       , **argv, required=False, type=str, help="output file prefix (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading structures from file '{args.input}'"):
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    Ns = len(structures)
    print(f"\t n. of structures: {Ns}\n")
    
    #------------------#
    
    arange = np.arange(Ns)
    indices_without_nan = np.full(Ns,True,dtype=bool)
    indices_with_nan = np.full(Ns,False,dtype=bool)
    
    keys = structures.get_keys()
    for key in keys:
        with eslog(f"Processing key '{key}'"):
            data = structures.get(key)
            data = np.reshape(data,(Ns,-1))
            nan_mask = np.any(np.isnan(data),axis=1)
            
            indices_without_nan = np.logical_and(indices_without_nan,~nan_mask)
            indices_with_nan = np.logical_or(indices_with_nan,nan_mask)
            
    assert np.sum(indices_without_nan)+np.sum(indices_with_nan) == Ns, "Error in the nan splitting!"
    assert np.all(np.logical_not(np.logical_and(indices_without_nan,indices_with_nan))), "Error in the nan splitting!"       
    
    #------------------#
    structures_without_nan = structures.subsample(arange[indices_without_nan])
    structures_with_nan = structures.subsample(arange[indices_with_nan])
    
    file_without_nan = f"{args.output}.wo-nan.extxyz"
    file_with_nan = f"{args.output}.nan.extxyz"
    
    #------------------#
    with eslog(f"Saving structures to file '{file_without_nan}'"):
        structures_without_nan.to_file(file=file_without_nan,format="extxyz")
        
    with eslog(f"Saving structures to file '{file_with_nan}'"):
        structures_with_nan.to_file(file=file_with_nan,format="extxyz")
            
    return
        
#---------------------------------------#
if __name__ == "__main__":
    main()

