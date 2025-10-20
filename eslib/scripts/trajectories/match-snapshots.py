#!/usr/bin/env python

import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning, everythingok


#---------------------------------------#
# Description of the script's purpose
description = "Find which snapshot of one trajectory correspond to the ones of another trajectory."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"    , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-r" , "--reference"       , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-rf", "--reference_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"          , **argv, required=False, type=str  , help="txt output file (default: %(default)s)", default='indices.txt')
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    structure = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tNumber of target snapshots: {:d}".format(len(structure)))
    # shape: (num_structures, num_atoms, 3)
    target_pos = structure.get("positions")
    
    #------------------#
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.reference), end="")
    reference = AtomicStructures.from_file(file=args.reference,format=args.reference_format)
    print("done")
    print("\tNumber of reference snapshots: {:d}".format(len(reference)))
    # shape: (num_ref, num_atoms, 3)
    ref_pos = reference.get("positions")
    
    #------------------#
    if len(reference) > len(structure):
        print(f"\n\t{warning}: the reference trajectory has more snapshots ({len(reference)}) than the target trajectory ({len(structure)})")
    
    #------------------#
    print("\n\tFinding matching snapshots ... ", end="")
    indices = np.zeros((len(target_pos),),dtype=int)
    for n,pos in enumerate(target_pos):
        # shape: (num_ref, num_atoms, 3)
        diff = ref_pos - pos[None,:,:]
        dist = np.linalg.norm(diff,axis=(1,2))
        idx = np.argmin(dist)
        indices[n] = idx  if dist[idx] < 1e-5 else -1
    print("done")
        
    #------------------#
    n_found = (indices >= 0).sum()
    if n_found != len(reference):
        print(f"\n\t{warning}: only {n_found} out of {len(reference)} snapshots were found in the reference trajectory")
    else:
        print(f"\n\t{everythingok}: all {len(reference)} snapshots were found in the reference trajectory")
        
    #------------------#
    print("\n\tWriting indices to output file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output,indices,fmt="%d")
    print("done")
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
