#!/usr/bin/env python
from eslib.formatting import esfmt, warning
from eslib.classes.mace_model import MACEModel
from eslib.functions import args_to_dict

#---------------------------------------#
description = "Allocate a MACE model and save it to a pickle file for later easier access."

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
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="*.pickle output file with the MACE model")
    parser.add_argument("-jo", "--json_output"  , **argv, type=str, required=False, help="JSON output file with all the previous instructions to initialize the MACE model (default: %(default)s)",default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\n\tAllocating the MACE model ... ",end="")
    kwargs = args_to_dict(args)
    del kwargs["output"]
    del kwargs["json_output"]
    model = MACEModel(**kwargs)
    print("done")

    #------------------#
    print("\n\tMACE model summary: ")
    model.summary(string="\t\t")

    #------------------#
    try:
        print("\n\tSaving the MACE model to file '{:s}' ... ".format(args.output),end="")
        model.to_pickle(args.output)
        print("done") 
    except:
        print("\n\t{:s}: a problem occurred by saving the model to file using `pickle`.".format(warning))
        if args.json_output is None:
            print("It's recommended to specify -jo,--json_output.")

    if args.json_output is not None:
        print("\n\tSaving the MACE model input arguments to file '{:s}' ... ".format(args.json_output),end="")
        model.to_json(args.json_output)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()