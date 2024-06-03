#!/usr/bin/env python
import json
from eslib.formatting import esfmt
from eslib.show import show_dict
from eslib.classes.dipole import DipoleMACECalculator
from eslib.functions import args_to_dict

#---------------------------------------#
description = "Create a dipole model from a MACE neural netowork."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-m" , "--model"        , **argv, type=str, required=True , help="*.pth file with the MACE model")
    parser.add_argument("-mt", "--model_type"   , **argv, type=str, required=True , help="model type")
    parser.add_argument("-d" , "--device"       , **argv, type=str, required=False, help="device (default: %(default)s)", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("-dt", "--default_dtype", **argv, type=str, required=False, help="default dtype (default: %(default)s)", choices=["float32", "float64"], default="float64")
    parser.add_argument("-bs", "--batch_size"   , **argv, type=int, required=False, help="batch size (default: %(default)s)", default=64)
    parser.add_argument("-ck", "--charges_key"  , **argv, type=str, required=False, help="key of atomic charges (default: %(default)s)", default="Qs")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="*.pickle output file with the dipole model")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\n\tCreating dipole model based on the MACE model ... ",end="")
    kwargs = args_to_dict(args)
    del kwargs["output"]
    model = DipoleMACECalculator(**kwargs)
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
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/nn/nn2model.py",
#             "cwd" : "/home/stoccoel/google-personal/works/oxn/",
#             "console": "integratedTerminal",
#             "args" : ["-m", "model/AtomicDipolesMACE_MTP-32x0e+32x1o.model","-mt","AtomicDipolesMACE_MTP","-o","model.pickle"],
#             "justMyCode": true,
#         }
#     ]
# }