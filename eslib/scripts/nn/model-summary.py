#!/usr/bin/env python
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
    parser.add_argument("-i" , "--input", **argv, type=str, required=True , help="input file with the model")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\tLoading model from file '{:s}' ... ".format(args.input), end="")
    model = eslibModel.from_pickle(file=args.input)
    print("done")
    model.summary()

    return 0     

#---------------------------------------#
if __name__ == "__main__":
    main()