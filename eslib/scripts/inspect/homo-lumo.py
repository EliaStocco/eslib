#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.io_tools import pattern2sorted_files

#---------------------------------------#
# Description of the script's purpose
description = "Read the HOMO-LUMO gap from a bunch of aims-output files."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, type=str, help="Glob pattern for FHI-aims output files")
    parser.add_argument("-o", "--output", **argv, required=True, type=str, help="Path to output text file")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    import re

    print(f"\tExtracting files from the input pattern '{args.input}' ... ", end="")
    files = pattern2sorted_files(args.input)
    print("done")
    if not files:
        raise FileNotFoundError(f"No files found matching the pattern: {args.input}")
    print("\tNumber of files found:", len(files))

    results = []

    print("\tReading HOMO-LUMO gaps:")
    for fpath in files:
        gap_value = None
        with open(fpath, 'r') as f:
            for line in f:
                if "HOMO-LUMO" in line:
                    gap_value = line.strip()  # keep updating with last match

        if gap_value:
            # Try to extract the number (e.g., from "HOMO-LUMO gap: 2.15 eV")
            match = re.search(r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)[ \t]*eV", gap_value)
            if match:
                value = float(match.group(1))
                results.append(value)
                print(f"\t{fpath}: {value:.6f} eV")
            else:
                print(f"\t{fpath}: HOMO-LUMO line found but value not parsed")
                results.append(np.nan)
        else:
            print(f"\t{fpath}: HOMO-LUMO line NOT found")
            results.append(np.nan)

    # Save all extracted values
    np.savetxt(args.output, results)
    print(f"\nSaved HOMO-LUMO gaps to '{args.output}'")

#---------------------------------------#
if __name__ == "__main__":
    main()
