#!/usr/bin/env python
import json
import numpy as np
from eslib.classes.models.dipole import DipolePartialCharges
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.show import show_dict
from eslib.tools import is_integer
from typing import List
from ase import Atoms

#---------------------------------------#
# Description of the script's purpose
description = "Add partial charges to atomic structures."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--name"          , **argv, required=False, type=str, help="name for the charges (default: %(default)s)", default='Qs')
    parser.add_argument("-c" , "--charges"       , **argv, required=True , type=str, help="JSON file with the charges")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file with the atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # charges
    print("\tReading the charges from file '{:s}' ... ".format(args.charges), end="")
    with open(args.charges, 'r') as json_file:
        charges:dict = json.load(json_file)
    print("done")

    #------------------#
    print("\n\tCharges: ")
    show_dict(charges,"\t\t",2)

    for k,c in charges.items():
        if not is_integer(c):
            print("\t{:s}: '{:s}' charge is not an integer".format(warning,k))
        charges[k] = np.round(c,0)

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #------------------#
    print("\n\tCreating dipole model based on the charges ... ",end="")
    model = DipolePartialCharges(charges)
    print("done")

    #------------------#
    print("\n\tAdding charges as '{:s}' to the 'arrays' of the atomic structures ... ".format(args.name),end="")
    for n,structure in enumerate(atoms):
        if not model.check_charge_neutrality(structure):
            raise ValueError("structure . {:d} is not charge neutral".format(n))
        structure.arrays[args.name] = model.get_all_charges(structure)
    print("done")
    
    #------------------#
    # summary
    print("\n\tSummary of the properties: ")
    df = atoms.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

    
    #------------------#
    print("\n\tWriting the atomic structures to file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()