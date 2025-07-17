#!/usr/bin/env python
import json
# from typing import List

import numpy as np
# from ase import Atoms

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models.dipole import DipolePartialCharges
from eslib.formatting import esfmt, warning
from eslib.show import show_dict
from eslib.tools import is_integer

#---------------------------------------#
# Description of the script's purpose
description = "Add Mulliken charges to atomic structures."

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
    return parser

import re
import os

def extract_mulliken_data(file_path):
    """Extract electrons and charge for each atom from Mulliken analysis."""
    data = []
    in_block = False
    with open(file_path, 'r') as f:
        for line in f:
            if "Summary of the per-atom charge analysis" in line:
                in_block = True
                continue
            if in_block:
                if re.match(r"\s*\|\s+Total", line):
                    break
                match = re.match(r"\s*\|\s+(\d+)\s+([\d\.\-Ee]+)\s+([\d\.\-Ee]+)", line)
                if match:
                    atom_index = int(match.group(1))
                    electrons = float(match.group(2))
                    charge = float(match.group(3))
                    data.append((atom_index, electrons, charge))
    return data

def process_all_files(directory):
    """Process all files in a directory and print extracted data."""
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            result = extract_mulliken_data(filepath)
            if result:
                print(f"\nFile: {filename}")
                for atom_index, electrons, charge in result:
                    print(f"Atom {atom_index}: Electrons = {electrons:.6f}, Charge = {charge:.6f}")

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
        # charges[k] = np.round(c,0)

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
    try:
        print("\n\tSummary of the properties: ")
        df = atoms.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t"))
    except:
        pass

    
    #------------------#
    print("\n\tWriting the atomic structures to file '{:s}' ... ".format(args.output), end="")
    atoms.to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()