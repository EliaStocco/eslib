#!/usr/bin/env python

import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning, everythingok


#---------------------------------------#
# Description of the script's purpose
description = "Merge unequal trajectories using the output of 'match-snapshots.py'."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"    , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-r" , "--reference"       , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-rf", "--reference_format", **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--indices"         , **argv, required=True , type=str, help="txt file with the indices produced by 'match-snapshots.py'")
    parser.add_argument("-o" , "--output"          , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of", "--output_format"   , **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    structure = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tNumber of target snapshots: {:d}".format(len(structure)))
    
    #------------------#
    # summary
    print("\n\tSummary of the structures: ",end="")
    df = structure.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    
    #------------------#
    print("\n\tReading atomic structure from input file '{:s}' ... ".format(args.reference), end="")
    reference = AtomicStructures.from_file(file=args.reference,format=args.reference_format)
    print("done")
    print("\tNumber of reference snapshots: {:d}".format(len(reference)))
    
    #------------------#
    # summary
    print("\n\tSummary of the reference structures: ",end="")
    df = reference.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    
    #------------------#
    if len(reference) > len(structure):
        print(f"\n\t{warning}: the reference trajectory has more snapshots ({len(reference)}) than the target trajectory ({len(structure)})")
    
    #------------------#
    print("\n\tReading indices from file '{:s}' ... ".format(args.indices), end="")
    indices = np.loadtxt(args.indices,dtype=int)
    print("done")
    
    ref_indices = indices[indices>=0]
    assert len(ref_indices) == len(reference), f"Number of found indices ({len(ref_indices)}) does not match number of reference snapshots ({len(reference)})"
    assert len(indices) == len(structure), f"Number of indices ({len(indices)}) does not match number of reference snapshots ({len(structure)})"
    
    #------------------#
    print("\n\tPreparing data ... ", end="")
    merged = [None]
    data = reference.infoarrays2dict(remove=["numbers"])
    print("done")
    
    #------------------#
    print("\tMatching data ... ", end="")
    final_data = {}
    final_what = {}
    for key,all_value in data.items():
        ncols = all_value["ncols"]
        what = all_value["what"]
        all_value = all_value["data"]
        
        # just for debugging
        okey = "new_positions" if key == "positions" else key
        
        if what == "info":
            final_what[okey] = "info"
            tmp = np.asarray(all_value[0])
            if tmp.ndim == 0 or tmp.shape[0] == 1:
                final_data[okey] = [np.nan]*len(structure)
            elif tmp.ndim != 1:
                raise ValueError(f"Cannot handle info '{key}' with shape {tmp.shape}")
            else:
                final_data[okey] = [np.zeros(tmp.shape[0])]*len(structure)
                
            final_data[okey] = np.asarray(final_data[key])
            final_data[okey][indices>=0] = all_value[ref_indices]
            final_data[okey][indices<0] = np.nan
        
        else:
            final_what[okey] = "arrays"
            final_data[okey] = [None]*len(structure)
            for n in range(len(structure)):
                ii = indices[n]
                Natoms = structure[n].get_global_number_of_atoms()
                if ii >= 0 :
                    final_data[okey][n] = all_value[ii] 
                else:
                    final_data[okey][n] = np.full((Natoms,ncols),np.nan)
    print("done")

    #------------------#
    print("\tMerging trajectories ... ", end="")
    for key,value in final_data.items():
        structure.set(key,value,final_what[key])
    print("done")
    
    #------------------#
    # summary
    print("\n\tSummary of the merged structures: ",end="")
    df = structure.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    
    #------------------#
    # debugging
    if structure.has("new_positions"): 
        new_pos = structure.get("new_positions")
        ii = ~np.isnan(new_pos).any(axis=(1,2))
        new_pos = new_pos[ii]
        pos = structure.get("positions")[ii]
        assert np.allclose(new_pos,pos,atol=1e-8), "Positions do not match after merging!"
        
    #------------------#
    print("\n\tWriting structure structures to file '{:s}' ... ".format(args.output), end="")
    structure.to_file(file=args.output,format=args.output_format)
    print("done")
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
