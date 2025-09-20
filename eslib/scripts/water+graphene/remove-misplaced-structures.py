#!/usr/bin/env python
import os
import numpy as np
from typing import List
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.mathematics import gaussian_cluster_indices
from concurrent.futures import ProcessPoolExecutor, as_completed

#---------------------------------------#
# Description of the script's purpose
description = "Remove confined water structures with hydrogen or oxygends laying outside the graphene layers."
documentation = "Please first run 'prettify-confined-water.py'."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    parser.add_argument("-j" , "--jobs"         , **argv, required=False, type=int  , help="number of parallel processes (default: %(default)s)", default=int(os.cpu_count()/2))
    return parser

#---------------------------------------#
def process_structure(atoms: Atoms) -> Atoms:
    """Process one atomic structure."""
    carbons = Atoms([a for a in atoms if a.symbol == "C"])
    others = Atoms([a for a in atoms if a.symbol != "C"])
    Cpos = carbons.get_positions()[:,2]
    z = gaussian_cluster_indices(Cpos, 2)
    assert len(z.keys()) == 2, "Not a graphene double layer."
    keys = list(z.keys())
    layer_1 = np.mean(Cpos[z[keys[0]]])
    layer_2 = np.mean(Cpos[z[keys[1]]])
    if layer_1 > layer_2:
        layer_1, layer_2 = layer_2, layer_1
    assert layer_1 < layer_2, "error"
    
    above = np.any(others.positions[:,2] > layer_2)
    below = np.any(others.positions[:,2] < layer_1)
    
    return not (above or below)

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):
    
    #-------------------#
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    structures:List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tn. of structures: ", len(structures))
    
    #-------------------#
    if args.jobs > 1:
        print(f"\tProcessing {len(structures)} structures in parallel with {args.jobs} workers ...",end="")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            ii = list(executor.map(process_structure, structures))
    else:
        print(f"\tProcessing {len(structures)} structures sequentially ...",end="")
        ii = [process_structure(atoms) for atoms in structures]
    print("done")
    ii = np.asarray(ii)
    print(f"\tKeeping {sum(ii)} structures")
    print(f"\tRemoving {sum(~ii)} structures")
    to_save = structures.subsample(np.where(ii)[0])

    #------------------#
    print("\n\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    to_save.to_file(file=args.output,format=args.output_format)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
