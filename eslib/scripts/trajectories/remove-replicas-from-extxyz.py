#!/usr/bin/env python
import re
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, eslog, message, warning
from eslib.input import ilist

#---------------------------------------#
# Description of the script's purpose
description = "Detect the replicas in a extxyz file."

# TODO: still to debug

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str  , help="input file [extxyz]")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-k"  , "--keyword"      , **argv, required=False, type=str  , help="keyword of the info/array (default: %(default)s)", default="ipi_comment")
    parser.add_argument("-s"  , "--start_stride" , **argv, required=False, type=ilist, help="start and stride, e.g. [1,4] (default: %(default)s)", default=[0,1])
    parser.add_argument("-o"  , "--output"       , **argv, required=False, type=str, help="output file with the unique indices/steps(default: %(default)s)", default=None)
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output format for np.savetxt (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading structures from file '{args.input}'"):
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print(f"\tn. of structures: {len(structures)}")
        
    #------------------#
    with eslog(f"Extracting '{args.keyword}'"):
        ipi_comments = structures.get(args.keyword)
        
    #------------------#
    with eslog(f"Extracting MD steps"):   
        steps = np.asarray([int(re.search(r'Step:\s+(\d+)', line).group(1)) for line in ipi_comments]).astype(int)
    
    suggested_stride = np.round(np.diff(steps).mean(),0).astype(int)
    print(f"\n\tThe snapshots seem to be sampled with a stride of: {suggested_stride}")
    if suggested_stride != args.start_stride[1]:
        print(f"\t{warning}: the suggested stride and the one that you provided do not match.")
    
    #------------------#
    with eslog(f"Looking for replicas and missing values"):   
        # Generate the target array
        to_keep = np.arange(args.start_stride[0], max(steps)+1, args.start_stride[1])

        # Get mask of which values in `to_keep` are missing from `steps`
        missing_mask = ~np.isin(to_keep, steps)

        # Get values and indices in `to_keep` that are missing
        missing_values = to_keep[missing_mask]
        # missing_indices_in_to_keep = np.where(missing_mask)[0]

        if len(missing_values) == 0 :
            
            u,indices = np.unique(steps,return_index=True)
            assert np.allclose(steps[indices],u), "coding error"

            # indices = np.where(np.isin(steps, to_keep))[0]
            # assert np.allclose(steps[indices],to_keep), "coding error"
        else:
            raise ValueError("not implemented yet")
    
    #------------------#
    with eslog(f"Subsampling"):   
        structures = structures.subsample(indices) 
        # ipi_comments = structures.get(args.keyword)
        # steps = np.asarray([int(re.search(r'Step:\s+(\d+)', line).group(1)) for line in ipi_comments]).astype(int)
        # assert np.allclose(steps,to_keep), "coding error"
    print(f"\tn. of structures: {len(structures)}")
    
    #------------------#
    with eslog(f"Saving structures to file '{args.output}'"):
        structures.to_file(file=args.output,format=args.output_format)
            
    return
        

#---------------------------------------#
if __name__ == "__main__":
    main()

