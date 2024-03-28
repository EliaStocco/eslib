#!/usr/bin/env python
import numpy as np
import json
from ase import Atoms
from eslib.classes.trajectory import trajectory as Trajectory
from eslib.formatting import esfmt
from eslib.show import show_dict
from eslib.classes.dipole import DipolePartialCharges

#---------------------------------------#
# Description of the script's purpose
description = "Create a Partial Charges model for the dipole from a BEC tensor."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-r", "--reference"      , **argv, type=str, required=True , help="file with a reference configuration [xyz]")
    parser.add_argument("-f", "--format"         , **argv, type=str, required=False, help="reference configuration format (default: 'None')" , default=None)
    parser.add_argument("-z", "--bec"            , **argv, type=str, required=True , help="file with a BEC tensor [txt]")
    parser.add_argument("-oc", "--output_charges", **argv, type=str, required=False, help="JSON output file with the partial charges (default: None)", default=None)
    parser.add_argument("-o" , "--output"        , **argv, type=str, required=False, help="pickle output file with the dipole model (default: 'DipolePC.pickle')", default="DipolePC.pickle")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structures from file '{:s}' ... ".format(args.reference), end="")
    reference:Atoms = Trajectory(args.reference,format=args.format,index=0)[0]
    print("done")

    #------------------#
    symbols = reference.get_chemical_symbols()
    species,index,inv = np.unique(symbols,return_index=True,return_inverse=True)
    print("\tAtoms: ",symbols)
    print("\tSpecies: ",species)

    #------------------#
    # trajectory
    print("\n\tReading the BEC tensor from file '{:s}' ... ".format(args.bec), end="")
    Z = np.loadtxt(args.bec)
    print("done")

    print("\tZ shape: ",Z.shape)

    assert Z.shape[0] == 3*reference.get_global_number_of_atoms()

    #------------------#
    print("\tComputing partial charges ... ",end="")
    charges = {}
    Z = Z.reshape((-1,3,3))
    for n,s in enumerate(species):
        ii = [ i for i,v in enumerate(inv) if v == n ] 
        Zi = Z[ii]
        _charges = np.zeros(len(Zi))
        for k,z in enumerate(Zi):
            _charges[k] = np.trace(z)/3
        charges[s] = _charges.mean()
    print("done")

    #------------------#
    print("\tPartial charges: ")
    show_dict(charges,"\t",2)

    #------------------#
    print("\tCreating dipole model based on the partial charges ... ",end="")
    model = DipolePartialCharges(charges)
    print("done")

    #------------------#
    print("\tSaving the dipole model to file '{:s}' ... ".format(args.output),end="")
    model.to_pickle(args.output)
    print("done")

    #------------------#
    if args.output_charges is not None:
        print("\tWriting partial charges to file '{:s}' ... ".format(args.output_charges), end="")
        with open(args.output_charges, 'w') as json_file:
            json.dump(charges, json_file, indent=4)
        print("done")    

#---------------------------------------#
if __name__ == "__main__":
    main()

# { 
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/convert/convert-file.py",
#             "cwd" : "/home/stoccoel/Downloads",
#             "console": "integratedTerminal",
#             "justMyCode": false,
#             "args" : ["-i", "i-pi.positions_0.xyz", "-if", "ipi", "-o", "water.extxyz","-sc","true"]
#         }
#     ]
# }