#!/usr/bin/env python
from ase import Atoms
from ase.io import write

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Create a linar model for the dipole of a system given the Born Effective Charges of a reference configuration."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
       
    print("\n\tWriting oxidation numbers as 'info' to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=atoms,filename=args.output) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    

#---------------------------------------#
if __name__ == "__main__":
    main()