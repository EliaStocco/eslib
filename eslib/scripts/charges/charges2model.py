#!/usr/bin/env python
import json

from eslib.classes.models.dipole import DipolePartialCharges
from eslib.formatting import esfmt
from eslib.input import str2bool
from eslib.show import show_dict

#---------------------------------------#
description = "Create a dipole model from partial charges."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-c", "--charges", **argv, type=str, required=True , help="JSON file with the partial charges")
    parser.add_argument("-z", "--compute_BEC", **argv, type=str2bool, required=False , help="whether to compute BEC (default: %(default)s)", default=False)
    parser.add_argument("-o", "--output" , **argv, type=str, required=False, help="pickle output file with the dipole model (default: %(default)s)", default="DipolePC.pickle")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the partial charges from file '{:s}' ... ".format(args.charges), end="")
    with open(args.charges, 'r') as json_file:
        charges = json.load(json_file)
    print("done")

    #------------------#
    print("\n\tPartial charges: ")
    show_dict(charges,"\t\t",2)

    #------------------#
    print("\n\tCreating dipole model based on the partial charges ... ",end="")
    model = DipolePartialCharges(charges=charges,compute_BEC=args.compute_BEC)
    print("done")

    #------------------#
    print("\n\tSaving the dipole model to file '{:s}' ... ".format(args.output),end="")
    model.to_pickle(args.output)
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