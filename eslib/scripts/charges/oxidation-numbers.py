#!/usr/bin/env python
from ase.io import write
import numpy as np
import json
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.formatting import matrix2str
from eslib.physics import oxidation_number

#---------------------------------------#
# Description of the script's purpose
description = "Compute the nominal oxidation numbers of an atomic structure."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--numbers"       , **argv, required=False, type=str, help="JSON file with the user-defined oxidation numbers (default: %(default)s)", default=None)
    parser.add_argument("-k" , "--keyword"       , **argv, required=False, type=str, help="keyword for the oxidation numbers (default: %(default)s)", default='oxidation-numbers')
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file with the oxidation numbers (default: %(default)s)", default="oxidation-numbers.extxyz")
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

    symbols = atoms.get_chemical_symbols()

    #------------------#
    numbers = None
    if args.numbers is not None:
        try:
            print("\tReading the user-defined oxidation numbers '{:s}' ... ".format(args.numbers), end="")
            numbers = json.loads(args.numbers)
            print("done")
        except:
            print("failded")
            print("\tReading the user-defined oxidation numbers  from file '{:s}' ... ".format(args.numbers), end="")
            with open(args.numbers, 'r') as f:
                numbers = json.load(f)
            print("done")

    #------------------#
    print("\tCompute the nominal oxidation number of the provided atoms ... ", end="")
    on = oxidation_number(symbols,numbers)
    print("done")

    #------------------#
    if not np.allclose(on.sum(),0):
        print("\t{:s}: the sum of the oxidation numbers is not zero --> subtracting the average.".format(warning))
        on_old = on.copy()
        mean = on.mean()
        print("\tSubtracted average: {:f}".format(mean))
        on -= mean
        M = np.concatenate([on_old.reshape((-1,1)),on.reshape((-1,1))], axis=1)
        col_names = ["Ox.","Corr."]
    else:
        M = on.reshape((-1,1))
        col_names = ["Ox."]

    if not np.allclose(on.sum(),0):
        raise ValueError("there is a bug in the code")
    
    print("\n\tComputed nominal oxidation numbers:")
    line = matrix2str(M,digits=2,col_names=col_names,cols_align="^",width=8,row_names=atoms.get_chemical_symbols())
    print(line)

    #------------------#    
    args.output = str(args.output)
    if args.output.endswith("txt"):
        print("\n\tWriting oxidation numbers to file '{:s}' ... ".format(args.output), end="")
        np.savetxt(args.output, on)
        print("done")
    else:       
        print("\n\tSaving oxidation numbers as '{:s}' into 'arrays' ... ".format(args.keyword), end="")
        atoms.arrays[args.keyword] = np.asarray(on).reshape((-1,1))
        print("done")

        print("\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
        try:
            write(images=atoms,filename=args.output) # fmt)
            print("done")
        except Exception as e:
            print("\n\tError: {:s}".format(e))

#---------------------------------------#
if __name__ == "__main__":
    main()