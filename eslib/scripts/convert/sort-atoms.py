#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
from eslib.classes.atomic_structures import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Sort the atoms (and their arrays) of a trajectory according to the provided indices."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv, required=True , type=str, help="input file with atomic structure")
    parser.add_argument("-if" , "--input_format" ,   **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n"  , "--indices"      ,   **argv, required=False, type=str, help="*.txt file with the indices")
    parser.add_argument("-o"  , "--output"       ,   **argv, required=True , type=str, help="output file")
    parser.add_argument("-of" , "--output_format",   **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading indices from file '{:s}' ... ".format(args.indices), end="")
    indices = np.loadtxt(args.indices,dtype=int).astype(int)
    print("done")
    
    #------------------#
    # trajectory
    print("\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(structures)))
    
    Natoms = len(structures[0])
    assert all(len(s) == Natoms for s in structures), "All the structures must have the same number of atoms"
    print("\tn. of atoms: {:d}".format(Natoms))
    
    #------------------#
    print("\n\tSorting the following arrays: ", structures.get_keys("arrays"))
    print("\tSorting: ")
    N = len(structures)
    for n,atoms in enumerate(structures):
        print("\t{:d}/{:d}".format(n+1,N), end="\r")
        for k in atoms.arrays.keys():
            atoms.arrays[k] = atoms.arrays[k][indices]
    print("\n\tdone")
    
    #------------------#
    print("\n\tWriting reference structure to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
