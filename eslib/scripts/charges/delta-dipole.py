#!/usr/bin/env python
import json
import numpy as np
from eslib.classes.dipole import DipolePartialCharges
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, warning
from eslib.show import show_dict
from eslib.tools import is_integer

#---------------------------------------#
# Description of the script's purpose
description = "Remove the point-charges contribution to the dipole."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="input file format (default: 'None')", default=None)
    parser.add_argument("-id", "--in_dipole"    , **argv, required=False, type=str, help="name for the input dipole (default: 'dipole')", default='dipole')
    parser.add_argument("-od", "--out_dipole"   , **argv, required=False, type=str, help="name for the output dipole (default: 'delta_dipole')", default='delta_dipole')
    parser.add_argument("-c" , "--charges"      , **argv, required=True , type=str, help="JSON file with the charges")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file with the atomic structures")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: 'None')", default=None)
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
    atoms:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")

    #------------------#
    print("\n\tCreating dipole model based on the charges ... ",end="")
    model = DipolePartialCharges(charges)
    print("done")

    #------------------#
    print("\n\tRemoving point-charges contribution to '{:s}' and saving it to '{:s}' ... ".format(args.in_dipole,args.out_dipole),end="")
    for n,structure in enumerate(atoms):
        if not model.check_charge_neutrality(structure):
            raise ValueError("structure . {:d} is not charge neutral".format(n))
        pc = model.get([structure])[0]
        dipole = structure.info[args.in_dipole]
        structure.info[args.out_dipole] = dipole - pc
    print("done")
    
    #------------------#
    print("\tWriting the atomic structures to file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(args.output,args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()