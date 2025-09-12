#!/usr/bin/env python
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.mathematics import gaussian_cluster_indices

#---------------------------------------#
description = "Compute the volumetric density of water within two graphene layers."
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file with one graphene layer only (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    return  parser


#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #-------------------#
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input, format=args.input_format, index=0)[0]
    print("done")
    
    #-------------------#
    print("\tSelecting graphene atoms ... ", end="")
    carbons = Atoms([a for a in atoms if a.symbol == "C"])
    print(f"done, found {len(carbons)} carbon atoms")
    
    #-------------------#
    print("\tDetecting graphene layers ... ", end="")
    Cpos = carbons.get_positions()[:,2]
    z = gaussian_cluster_indices(Cpos, 2)
    assert len(z.keys()) == 2, "Not a graphene double layer."
    keys = list(z.keys())
    layer_1 = np.mean(Cpos[z[keys[0]]])
    layer_2 = np.mean(Cpos[z[keys[1]]])
    if layer_1 > layer_2:
        layer_1, layer_2 = layer_2, layer_1
    print("done")
    print(f"\tFirst layer z ~ {layer_1:.2f}")
    print(f"\tSecond layer z ~ {layer_2:.2f}")
    
    #-------------------#
    print("\tAssigning atoms to layers ... ", end="")
    # Assign atoms to each layer using the GMM indices
    lower_layer = Atoms([carbons[i] for i in z[keys[0]]],cell=atoms.cell,pbc=atoms.pbc)
    upper_layer = Atoms([carbons[i] for i in z[keys[1]]],cell=atoms.cell,pbc=atoms.pbc)
    print("done")
    print(f"\tLower layer atoms: {len(lower_layer)}")
    print(f"\tUpper layer atoms: {len(upper_layer)}")
    
    #-------------------#
    print("\tSaving graphene layers to files ... ", end="")
    import os
    base, ext = os.path.splitext(args.output)
    lower_file = f"{base}_lower{ext}"
    upper_file = f"{base}_upper{ext}"

    lower_layer.write(lower_file, format=args.output_format)
    upper_layer.write(upper_file, format=args.output_format)
    print("done")
    print(f"\tLower layer saved to: {lower_file}")
    print(f"\tUpper layer saved to: {upper_file}")

#---------------------------------------#
if __name__ == "__main__":
    main()
