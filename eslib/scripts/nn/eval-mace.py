#!/usr/bin/env python
import json
from ase.io import write
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.classes.mace_model import MACEModel
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate a MACE model."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-m" , "--model"        , **argv, type=str, required=True , help="*.pth file with the MACE model of JSON file with instructions")
    parser.add_argument("-p" , "--prefix"       , **argv, type=str, required=False, help="prefix to be prepended to the properties evaluated by the MACE model (default: 'MACE_')", default="MACE_")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=False, help="output file with the atomic structures and the predicted properties (default: 'mace.extxyz')", default="mace.extxyz")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: 'None')", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if str(args.model).endswith(".json"):
        print("\tReading MACE model input parameters from file '{:s}' ... ".format(args.model), end="")
        with open(args.model, 'r') as json_file:
            kwargs = json.load(json_file)
        print("done")

        print("\tAllocating MACE model ... ", end="")
        model = MACEModel(**kwargs)
        print("done")
    else:
        print("\tLoading MACE model from file '{:s}' ... ".format(args.model), end="")
        model = MACEModel.from_pickle(args.model)
        print("done")

    #------------------#
    print("\n\tMACE model summary: ")
    model.summary(string="\t\t")

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(structures)))

    #------------------#
    print("\n\tEvaluating the MACE model ... ", end="")
    output = model.compute(structures,args.prefix)
    print("done")

    #------------------#
    print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.output), end="")
    output.to_file(args.output,args.output_format)
    print("done")
     

#---------------------------------------#
if __name__ == "__main__":
    main()