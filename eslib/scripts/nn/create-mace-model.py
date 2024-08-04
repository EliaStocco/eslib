#!/usr/bin/env python
from eslib.formatting import esfmt
from eslib.classes.models.mace_model import MACEModel
from eslib.classes.models.dipole import DipoleMACECalculator
from mace.calculators import MACECalculator
from eslib.functions import args_to_dict
from eslib.input import slist, str2bool
import json

#---------------------------------------#
description = "Allocate a MACE model and save it to a pickle file for later easier access."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=False, help="JSON file with all the following parameters default: %(default)s)", default=None)
    parser.add_argument("-m" , "--model_path"   , **argv, type=str, required=True , help="*.pth file with the MACE model")
    parser.add_argument("-mt", "--model_type"   , **argv, type=str, required=True , help="eslib model type (default: %(default)s)", choices=["MACEModel", "DipoleMACECalculator"], default="MACEModel")
    parser.add_argument("-d" , "--device"       , **argv, type=str, required=False, help="device (default: %(default)s)", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("-dt", "--default_dtype", **argv, type=str, required=False, help="default dtype (default: %(default)s)", choices=["float32", "float64"], default="float64")
    parser.add_argument("-bs", "--batch_size"   , **argv, type=int, required=False, help="batch size (default: %(default)s)", default=64)
    parser.add_argument("-ck", "--charges_key"  , **argv, type=str, required=False, help="key of atomic charges (default: %(default)s)", default="Qs")
    parser.add_argument("-dR", "--dR"           , **argv, type=str2bool, required=False, help="whether to compute spatial derivatives (default: %(default)s)", default=False)
    parser.add_argument("-dp", "--to_diff_props"  , **argv, type=slist, required=False, help="properties to be differentiated (default: %(default)s)", default=["dipole"])
    parser.add_argument("-rp", "--rename_props"  , **argv, type=dict, required=False, help="properties to be renamed (default: %(default)s)", default={"dipole_dR":"BEC"})
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="*.pickle output file with the MACE model")
    # dR:bool
    # to_diff_props:List
    # rename_dR:Dict[str,Any]
    # parser.add_argument("-jo", "--json_output"  , **argv, type=str, required=False, help="JSON output file with all the previous instructions to initialize the MACE model (default: %(default)s)",default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    if args.input is not None:
        print("\tReading parameters from file '{:s}' ... ".format(args.input),end="")
        with open(args.input, 'r') as f:
            kwargs = json.load(f)
        print("done")
    else:
        kwargs = args_to_dict(args)

    print("\n\tUsing the following class: {:s}".format(args.model_type))
    if args.model_type == "MACEModel":
        mtype = MACEModel
    elif args.model_type == "DipoleMACECalculator":
        mtype = DipoleMACECalculator
    elif args.model_type == "MACECalculator":
        mtype = MACECalculator
    else:
        raise ValueError("Unknown model type '{:s}'.".format(args.model_type))
    
    if "input" in kwargs: del kwargs["input"]
    if "output" in kwargs: del kwargs["output"]
    if "model_type" in kwargs: del kwargs["model_type"]

    #------------------#
    print("\n\tInput for the MACE model:")
    for k, v in kwargs.items():
        max_key_length = max(len(key) for key in kwargs.keys())
        # Align the output based on the length of the longest key
        print("\t\t{:<{width}}: {}".format(k, v, width=max_key_length))   

    #------------------#
    print("\n\tAllocating the MACE model ... ",end="")
    model = mtype(**kwargs)
    print("done")

    if model is None:
        raise ValueError("Loaded model is None.")

    #------------------#
    try:
        model.summary()
    except:
        pass

    #------------------#
    # try:
    print("\n\tSaving the MACE model to file '{:s}' ... ".format(args.output),end="")
    model.to_pickle(args.output)
    print("done") 
    # except:
    #     print("\n\t{:s}: a problem occurred by saving the model to file using `pickle`.".format(warning))
    #     if args.json_output is None:
    #         print("It's recommended to specify -jo,--json_output.")

    # if args.json_output is not None:
    #     print("\n\tSaving the MACE model input arguments to file '{:s}' ... ".format(args.json_output),end="")
    #     model.to_json(args.json_output)
    #     print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()