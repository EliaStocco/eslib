#!/usr/bin/env python
import numpy as np
import json
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.show import show_dict

#---------------------------------------#
description = "Compute the partial charges from a BEC tensor."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-r", "--reference", **argv, type=str, required=True , help="file with a reference configuration [xyz]")
    parser.add_argument("-f", "--format"   , **argv, type=str, required=False, help="reference configuration format (default: %(default)s)" , default=None)
    parser.add_argument("-z", "--bec"      , **argv, type=str, required=True , help="file with a BEC tensor [txt]")
    parser.add_argument("-o", "--output"   , **argv, type=str, required=False, help="JSON output file with the partial charges (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structures from file '{:s}' ... ".format(args.reference), end="")
    reference:Atoms = AtomicStructures.from_file(file=args.reference,format=args.format,index=0)[0]
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
    print("\n\tPartial charges: ")
    show_dict(charges,"\t\t",2)

    #------------------#
    all_charges = [ charges[s] for s in reference.get_chemical_symbols() ]
    print("\n\tTotal charge: ",np.sum(all_charges))
    mean = np.mean(all_charges)
    for k in charges.keys():
        charges[k] -= mean

    #------------------#
    print("\n\tPartial charges (corrected): ")
    show_dict(charges,"\t\t",2)

    all_charges = [ charges[s] for s in reference.get_chemical_symbols() ]
    if np.sum(all_charges) > 1e-12:
        raise ValueError("coding error")

    # #------------------#
    # print("\tCreating dipole model based on the partial charges ... ",end="")
    # model = DipolePartialCharges(charges)
    # print("done")

    # #------------------#
    # print("\tSaving the dipole model to file '{:s}' ... ".format(args.output),end="")
    # model.to_pickle(args.output)
    # print("done")

    #------------------#
    if args.output is not None:
        print("\n\tWriting partial charges to file '{:s}' ... ".format(args.output), end="")
        with open(args.output, 'w') as json_file:
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
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/dipole/bec2charges.py",
#             "cwd" : "/home/stoccoel/google-personal/works/LiNbO3",
#             "console": "integratedTerminal",
#             "justMyCode": false,
#             "args" : ["-r", "vib/start.au.xyz", "-o", "vib/partial-charges.json","-z","vib/BEC.txt"]
#         }
#     ]
# }
