#!/usr/bin/env python
import argparse
import pickle
import numpy as np
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Inspect a pickle file to determine the data type/class name and data dimensions."

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", type=str, required=True, help="Pickle file to inspect")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\tInspecting file '{:s}' ... ".format(args.input), end="")
    with open(args.input, 'rb') as f:
        obj = pickle.load(f)
        obj_type = type(obj)
        
        info = {
            "Type/Class Name": obj_type.__name__,
            "Module": obj_type.__module__,
        }
        
        if isinstance(obj, (np.ndarray, list, dict, tuple)):
            if isinstance(obj, np.ndarray):
                info["Data Dimensions"] = obj.shape
            elif isinstance(obj, (list, tuple)):
                info["Length"] = len(obj)
            elif isinstance(obj, dict):
                info["Number of Keys"] = len(obj)
    print("done")

    print("\n\tExtracted information:")
    for key, value in info.items():
        value = str(value)
        print("\t - {:s}: {:s}".format(key, value))
    
#---------------------------------------#
if __name__ == "__main__":
    main()
