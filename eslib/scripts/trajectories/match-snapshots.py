#!/usr/bin/env python
import os
import numpy as np
from eslib.mathematics import find_duplicates
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning, everythingok, error


#---------------------------------------#
# Description of the script's purpose
description = "Find which snapshot of one trajectory correspond to the ones of another trajectory."
documentation = \
"If you provide the same file for both '--input' and '--reference', the code will check if the file has some duplicate structures.\n" + \
"In general, the file specified by '--output' will contain an integer number for each structure in '--input':\n" + \
" - the integer will be -1 if no structure in '--reference' is found.\n" + \
" - otherwise, the integer will be the index of the matching structure in '--reference'." 

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"    , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-r" , "--reference"       , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-rf", "--reference_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-d" , "--duplicate"       , **argv, required=False, type=str  , help="file with the duplicate indices (default: %(default)s)", default='duplicates.txt')
    parser.add_argument("-u" , "--unique"          , **argv, required=False, type=str  , help="file with the unique indices (default: %(default)s)", default='unique.txt')
    parser.add_argument("-t" , "--tolerance"       , **argv, required=False, type=float, help="tolerance (default: %(default)s)", default=1e-8)
    parser.add_argument("-o" , "--output"          , **argv, required=False, type=str  , help="txt output file (default: %(default)s)", default='indices.txt')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
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
        indices[n] = idx  if dist[idx] < args.tolerance else -1
    print("done")
    
    #------------------#
    if os.path.exists(args.duplicate):
        os.remove(args.duplicate)
        print(f"\tRemoved file '{args.duplicate}'")
    
    reverted_indices = {}
    to_remove = []
    for n in range(len(reference)):
        reverted_indices[n] = np.where(indices == n)[0].tolist()
        if len(reverted_indices[n]) > 1:
            locs = np.asarray(reverted_indices[n])
            to_remove.append(locs[1:].tolist())
            pos = target_pos[locs]
            diff = pos[:, None, :, :] - pos[None, :, :, :]
            dist = np.linalg.norm(diff, axis=(2, 3))
            # Get upper-triangular indices, excluding diagonal
            i, j = np.triu_indices(dist.shape[0], k=1)
            upper_values = list(dist[i, j])
            
            print(f"\t{warning}: reference {n} matched by inputs {reverted_indices[n]} (distances: {upper_values})")
            
            # Write upper_values to file
            with open(args.duplicate, "a") as f:
                f.write(f"{n}: {reverted_indices[n]}\n")
    
    #------------------#
    # Flatten to_remove (list of lists) and convert to set
    to_remove = set(x for sublist in to_remove for x in sublist)

    # Get all indices
    all_indices = set(range(len(structure)))

    # Remove the indices in to_remove
    unique = all_indices - to_remove

    # Convert to sorted NumPy array
    unique = np.array(sorted(unique))

    # Write unique indices to file
    print(f"\n\tWriting the indices of the unique structures to '{args.unique}' ... ", end="")
    np.savetxt(args.unique, unique, fmt="%d")
    print("done")
        
    #------------------#
    n_found = (indices >= 0).sum()
    if n_found < len(reference):
        print(f"\n\t{warning}: only {n_found} out of {len(reference)} snapshots were found in the reference trajectory.")
    elif n_found > len(reference):
        string = f"\n\t{error}: {n_found} out of {len(reference)} snapshots were found in the reference trajectory (there might duplicates in {args.input}).\n\t" \
            + f"The indices of the duplicate structures have been written to '{args.duplicate}'\n\t" \
            + f"and the indices of the unique structures have been written to '{args.unique}'."
        print(string)
    else:
        print(f"\n\t{everythingok}: all {len(reference)} snapshots were found in the reference trajectory.")
        
    #------------------#
    print("\n\tWriting indices to output file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output,indices,fmt="%d")
    print("done")
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
