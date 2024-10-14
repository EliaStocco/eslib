#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import convert
import numpy as np

#---------------------------------------#
# Description of the script's purpose
description = "Displace an atomic structure along all the cartesian coordinates: useful for replay mode in i-PI."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with the reference atomic structure [au]")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-pu", "--pos_unit"     , **argv, required=False, type=str, help="positions unit (default: %(default)s)" , default="atomic_unit")
    parser.add_argument("-d" , "--displacement" , **argv, required=False, type=float, help="displacement [au] (default: %(default)s)" , default=0.001)
    parser.add_argument("-du", "--displ_unit"   , **argv, required=False, type=str, help="displacement unit (default: %(default)s)" , default="atomic_unit")
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str  , help="output file with the displaced atomic structures (default: %(default)s)", default='replay.xyz')
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.displacement <= 0:
        raise ValueError("The displacement has to be positive.")
    factor = convert(1,"length",args.displ_unit,args.pos_unit)
    print("\n\tConverting displacement from '{:s}' to '{:s}'".format(args.displ_unit,args.pos_unit))
    print("\t{:f} {:s} = {:f} {:s}".format(args.displacement,args.displ_unit,args.displacement*factor,args.pos_unit))
    args.displacement *= factor
        
    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    # displaced structures
    print("\tCreating the displaced structures ... ", end="")
    positions = atoms.get_positions()
    N = positions.shape[0] * positions.shape[1]
    structures = [None]*(2*N)
    displ = np.zeros(N)
    k = 0 
    for n in range(N):
        for sign in [1,-1]:
            structures[k] = atoms.copy()
            if not isinstance(structures[n],Atoms):
                raise TypeError("wrong type")
            displ.fill(0)
            displ[n] = sign * args.displacement
            structures[k].set_positions(positions+displ.reshape(-1,3))
            k += 1
    print("done")
    
    #------------------#
    print("\n\tWriting the displaced structures to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=structures,filename=args.output,format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    
#---------------------------------------#
if __name__ == "__main__":
    main()