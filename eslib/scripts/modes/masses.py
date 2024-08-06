#!/usr/bin/env python
from ase import Atoms
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
import numpy as np
from ipi.utils.units import Elements
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Create a file with the masses of the dof used by i-PI."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv, required=True , type=str, help="input file with an atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output", **argv, required=False, type=str, help="output file with the masses [au]", default='i-pi.masses')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structure
    print("\tReading the first atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    # chemical symbols
    symbols = atoms.get_chemical_symbols()
    print("\tAtomic symbols: ",end="")
    print("[",end="")
    N = len(symbols)
    for n,s in enumerate(symbols):
        if n < N-1:
            print(" '{:2}',".format(s),end="")
        else:
            print(" '{:2}'".format(s),end="")
    print("]")

    #------------------#
    # masses
    masses = [ Elements.mass_list[s] for s in symbols ]
    print("\tMasses [dalton]: ",end="")
    print("[",end="")
    N = len(masses)
    for n,s in enumerate(masses):
        if n < N-1:
            print(" '{:2}',".format(s),end="")
        else:
            print(" '{:2}'".format(s),end="")
    print("]")

    #------------------#
    # convert
    print("\tConvetting the masses from 'dalton' to 'atomic_unit' ... ",end="")
    factor = convert(1,"mass","dalton","atomic_unit")   
    masses = [mass for mass in masses for _ in range(3)]
    masses = np.asarray(masses)* factor
    print("done")

    #------------------#
    print("\n\tWriting the masses to file '{:s}' ... ".format(args.output), end="")
    try:
        np.savetxt(args.output,masses,fmt=float_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))


#---------------------------------------#
if __name__ == "__main__":
    main()