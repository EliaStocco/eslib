#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import ilist

#---------------------------------------#
description = "Remove structures from an extxyz file by a list of indices."

def prepare_args(description):
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, type=str, required=True, help="Input extxyz file")
    parser.add_argument("-r", "--remove_indices", **argv, type=ilist, required=True,
                        help="Text file containing indices to remove (one index per line, zero-based)")
    parser.add_argument("-o", "--output", **argv, type=str, required=True, help="Output extxyz file")
    parser.add_argument("-of", "--output_format", **argv, type=str, default=None, required=False,
                        help="Output file format (default: extxyz)")
    return parser

@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    atoms = AtomicStructures.from_file(file=args.input, format="extxyz")
    N = len(atoms)
    print("done")

    # #------------------#
    # print(f"\tReading indices to remove from '{args.remove_indices}' ... ", end="")
    # with open(args.remove_indices, "r") as f:
    #     lines = f.readlines()
    # remove_indices = sorted(set(int(line.strip()) for line in lines if line.strip().isdigit()))
    # print(f"done (will remove {len(remove_indices)} structures)")

    # Validate indices
    invalid = [idx for idx in args.remove_indices if idx < 0 or idx >= N]
    if invalid:
        raise ValueError(f"Invalid indices found (out of bounds): {invalid}")

    #------------------#
    print("\tRemoving specified structures ... ", end="")
    keep_indices = [i for i in range(N) if i not in args.remove_indices]
    filtered_atoms = atoms.subsample(keep_indices)
    print(f"done (kept {len(keep_indices)} structures)")

    #------------------#
    print(f"\tWriting filtered structures to file '{args.output}' ... ", end="")
    filtered_atoms.to_file(file=args.output, format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
