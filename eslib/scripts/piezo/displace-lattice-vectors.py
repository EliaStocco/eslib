#!/usr/bin/env python
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Displace the lattice vectors of an atomic structures (with rotations included)."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-d" , "--displacement" , **argv, required=False, type=float, help="displacement [Å] (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    if args.displacement is None:
        print("\n\tThe displacement will be automatically chosen to increase the volume by 0.1%")
        delta = 0.001*3.  # 0.1%
        volume = structure.get_volume()
        args.displacement = np.cbrt(volume*(1+delta)) - np.cbrt(volume) 
        orig_volume = volume
        max_vol = volume*(1+delta/3.0)
        print(f"\tOriginal volume: {orig_volume} [ang3]")
        print(f"\tChosen displacement [ang]: {args.displacement:.6f}")
    else:
        orig_volume = structure.get_volume()
        print(f"\tOriginal volume: {orig_volume} [ang3]")
        max_vol = None

    #------------------#
    # the first structure is the unperturbed one
    displaced = [structure.copy() for _ in range(19) ]
    
    print("\n\tDisplacing structures:")
    i = 1
    for a in range(3): # cycle over lattice vectors
        for xyz in range(3): # cycle over directions
            for pm in [1.,-1.]: # positive and negative displacement
                cell = np.asarray(displaced[i].get_cell())
                cell[a,xyz] += pm*args.displacement
                displaced[i].set_cell(cell)
                volume = displaced[i].get_volume()
                frac = (volume-orig_volume) / orig_volume *100
                print("\t"+f" - {i:3d}) volume = {volume:.6} ang3 which is"+f" {frac:10.6}% larger than the original.")
                i += 1
                
    displaced = AtomicStructures(displaced)
    
    #------------------#
    print("\n\tWriting displaced atomic structures to file '{:s}' ... ".format(args.output), end="")
    displaced.to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()