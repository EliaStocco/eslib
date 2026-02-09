#!/usr/bin/env python
import json
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Summary of an MD trajectory."

def shape_or_dtype(x):
    """
    Return the shape of x if it is a NumPy array,
    otherwise return a simple data type name.
    """
    if isinstance(x, np.ndarray):
        return list(x.shape)   # JSON-friendly
    elif isinstance(x, np.generic):  # NumPy scalar
        return type(x.item()).__name__
    else:
        return type(x).__name__

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, type=str, required=True , help="input file")
    parser.add_argument("-if", "--input_format", **argv, type=str, required=False, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output" , **argv, type=str, required=False, help="JSON output file (default: %(default)s)",default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done\n")
    print(f"\tn. of atomic structures: {len(structures)}")
    
    #------------------#
    print("\n\tExtracting properties ... ",end="")
    all_infos:dict[str,list] = {}
    all_arrays:dict[str,list] = {}
    for atom in structures:
        infos = list(atom.info.keys())
        arrays = list(atom.arrays.keys())
        
        for i in infos:
            v = shape_or_dtype(atom.info[i])
            if i not in all_infos:
                all_infos[i] = [v]
            elif v not in all_infos[i]:
                all_infos[i].append(i)
        del i
                
        for a in arrays:
            v = shape_or_dtype(atom.arrays[a])
            if a not in all_arrays:
                all_arrays[a] = [v]
            elif v not in all_arrays[a]:
                all_arrays[a].append(a)
        del a
    print("done")
      
    #------------------#
    print("\n\tInfo summary:")  
    for k,v in all_infos.items():
        print(f"\t - {k}: ",v)

    print("\n\tArrays summary:")  
    for k,v in all_arrays.items():
        print(f"\t - {k}: ",v)
        
    summary = {
        "description": description,
        "n_structures": len(structures),
        "info": all_infos,
        "arrays": all_arrays,
    }

    #------------------#
    if args.output is not None:
        print(f"\n\tSaving summary to '{args.output}' ... ", end="")
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=4)
        print("done")
        
    return

#---------------------------------------#
if __name__ == "__main__":
    main()
