#!/usr/bin/env python
import os
import numpy as np
from typing import List
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.mathematics import gaussian_cluster_indices
from concurrent.futures import ProcessPoolExecutor

#---------------------------------------#
# Description of the script's purpose
description = "Prettify a confined water structure."
documentation = "Please first run 'fold.py' and then 'fix-bonds.py'."

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
    Cpos = carbons.get_positions()[:,2]
    z = gaussian_cluster_indices(Cpos, 2)
    assert len(z.keys()) == 2, "Not a graphene double layer."
    keys = list(z.keys())
    layer_1 = np.mean(Cpos[z[keys[0]]])
    layer_2 = np.mean(Cpos[z[keys[1]]])
    if layer_1 > layer_2:
        layer_1, layer_2 = layer_2, layer_1

    delta = layer_2 - layer_1
    height = atoms.cell.cellpar()[2]
    if delta > height/2.:
        if abs(height-layer_2) < layer_1:
            layer_to_move = layer_2
        else:
            layer_to_move = layer_1

        ii = z[keys[np.argmin(abs(keys-layer_to_move))]]
        atoms.positions[ii,2] -= height
        
        atoms.positions[:,2] += height/2.

    return atoms

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):
    
    #-------------------#
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    structures:List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    
    #-------------------#
    if args.jobs > 1:
        print(f"\tProcessing {len(structures)} structures in parallel with {args.jobs} workers ...",end="")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(process_structure, structures))
        structures = AtomicStructures(results)
    else:
        print(f"\tProcessing {len(structures)} structures sequentially ...",end="")
        results = [process_structure(atoms) for atoms in structures]
        structures = AtomicStructures(results)
    print("done")

    #------------------#
    print("\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
