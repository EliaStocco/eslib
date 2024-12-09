#!/usr/bin/env python
import json
from typing import Any, Dict
from eslib.classes.models import eslibModel
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Show information about a model."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"            , **argv, type=str, required=True , help="input file with the model")
    parser.add_argument("-op", "--output_properties", **argv, type=str, required=False, help="JSON output file with the implemented properties (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
def write_implemented_properties(file:str,properties:Dict[str,Any])->None:
    # Convert tuples to lists (JSON doesn't support tuples)
    data = {}
    for k, v in properties.items():
        if isinstance(v,tuple) and isinstance(v[0],type):
            data[k] = [v[0].__name__, *v[1:] ]
        else:
            data[k] = list(v)
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\tLoading model from file '{:s}' ... ".format(args.input), end="")
    model = eslibModel.from_pickle(file=args.input)
    print("done")
    model.summary()
    
    if args.output_properties is not None:
        print("\n\tWriting implemented properties to file '{:s}' ... ".format(args.output_properties), end="")
        write_implemented_properties(args.output_properties,model.implemented_properties)
        print("done")

    return 0     

#---------------------------------------#
if __name__ == "__main__":
    main()