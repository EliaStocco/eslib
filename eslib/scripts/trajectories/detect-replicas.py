#!/usr/bin/env python
import re
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, eslog, message, warning

#---------------------------------------#
# Description of the script's purpose
description = "Detect the replicas in a extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str, help="input file [extxyz]")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-k"  , "--keyword"      , **argv, required=False, type=str, help="keyword of the info/array (default: %(default)s)", default="ipi_comment")
    parser.add_argument("-s"  , "--steps"        , **argv, required=False, type=str, help="output file with all the steps read from the input file (default: %(default)s)", default="None")
    parser.add_argument("-u"  , "--unique"       , **argv, required=False, type=str, help="output file with the indices of the unique steps for subsampling (default: %(default)s)", default="indices.txt")
    parser.add_argument("-m"  , "--missing"      , **argv, required=False, type=str, help="output file with the missing steps (default: %(default)s)", default="missing.txt")
    parser.add_argument("-o"  , "--output"       , **argv, required=False, type=str, help="output file with the unique indices/steps(default: %(default)s)", default=None)
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output format for np.savetxt (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading structures from file '{args.input}'"):
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
        
    #------------------#
    with eslog(f"Extracting '{args.keyword}'"):
        ipi_comments = structures.get(args.keyword)
        
    #------------------#
    with eslog(f"Extracting indices"):   
        steps = np.asarray([int(re.search(r'Step:\s+(\d+)', line).group(1)) for line in ipi_comments]).astype(int)
        
    #------------------#
    if args.steps is not None:
        with eslog(f"Saving indices to file '{args.steps}'"):
            np.savetxt(args.steps,steps)
    
    #------------------#        
    all_steps = np.arange(np.max(steps) + 1, dtype=int)
    usteps, indices = np.unique(steps, return_index=True)

    if steps.shape == all_steps.shape and np.allclose(steps, all_steps):
        msg = "no replicas found"
        print(f"\n\t {message}: {msg}.") 
    else:
        msg = "found replicas"
        print(f"\n\t {warning}: {msg}.") 
        
        # usteps, indices = np.unique(steps, return_index=True)
        # assert np.allclose(steps,usteps[indices]), "coding error"
        # if msg == "no replicas found":
        #     assert np.allclose(usteps,steps), "coding error"
        #     assert np.allclose(usteps,indices), "coding error"
        
        with eslog(f"Saving unique indices to file '{args.unique}'"):
            np.savetxt(args.unique,indices)
            
        if args.output is not None:
            with eslog("Subsampling structures'"):
                structures  = structures.subsample(indices)
            with eslog(f"Saving structures to file '{args.output}'"):
                structures.to_file(file=args.output,format=args.output_format)
        
    #------------------#
    all_steps = np.arange(np.max(usteps) + 1, dtype=int)
    if usteps.shape == all_steps.shape and np.allclose(usteps, all_steps):
        msg = "no missing values found"
        print(f"\n\t {message}: {msg}.") 
    else:
        msg = "missing values"
        print(f"\n\t {warning}: {msg}.") 
        
        missing = np.setdiff1d(all_steps,usteps)
        
        with eslog(f"Saving missing steps to file '{args.unique}'"):
            np.savetxt(args.missing,missing)
            
    return
        

#---------------------------------------#
if __name__ == "__main__":
    main()

