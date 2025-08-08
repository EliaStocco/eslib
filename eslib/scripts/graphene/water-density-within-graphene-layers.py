#!/usr/bin/env python
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
from eslib.mathematics import group_floats_by_decimals
from eslib.physics import compute_density

#---------------------------------------#
description = "Compute the volumetric density of water within two graphene layers."
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=False, help="output file with water only (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    return  parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #-------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    #-------------------#
    print("\tDividing atoms ... ",end="")
    carbons = Atoms([ a for a in atoms if a.symbol == "C" ])
    water = Atoms([ a for a in atoms if a.symbol in ["O","H"] ])
    print("done")
    
    #-------------------#
    print("\tDetecting layers ... ",end="")
    Cpos = carbons.get_positions()[:,2]
    z = group_floats_by_decimals(Cpos,0)
    assert len(z.keys()) == 2, "you are not provided a graphene double layer."
    keys = list(z.keys())
    layer_1 = np.mean(z[keys[0]])
    layer_2 = np.mean(z[keys[1]]) 
    if layer_1 > layer_2:
        tmp = layer_2
        layer_2 = layer_1
        layer_1 = tmp
    print("done")
    print(f"\tFirst layer: {layer_1:.2f}")
    print(f"\tSecond layer: {layer_2:.2f}")
    z = layer_2 - layer_1
    print(f"\tDistance between layers: {z:.2f}")
    
    #-------------------#
    print("\tCreating water-only structure ... ",end="")
    cellpar = atoms.cell.cellpar()
    assert np.allclose(cellpar[3:],90), "Angles are not 90 deg."
    
    new_cellpar = np.asarray([cellpar[0],cellpar[1],layer_2-layer_1,90,90,90])
    water.set_cell(new_cellpar)
    water.set_center_of_mass(new_cellpar[:3]/2.)
    print("done")
    
    if args.output is not None:
        print(f"\tSaving water only structure to file '{args.output}' ... ",end="")
        AtomicStructures([water]).to_file(file=args.output,format=args.output_format)
        print("done")
        
    #-------------------#
    print("\tComputing density ... ",end="")
    density = compute_density(water)
    print("done")
    print("\tdensity [g/cm^3]: ",density)

#---------------------------------------#
if __name__ == "__main__":
    main()
