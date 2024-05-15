#!/usr/bin/env python
from eslib.classes.trajectory import AtomicStructures
from eslib.classes.dipole import DipoleModel
from eslib.formatting import esfmt
import numpy as np

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
    parser.add_argument("-m" , "--model"        , **argv, type=str, required=True , help="*.pickle file with the dioole model")
    parser.add_argument("-n" , "--name"         , **argv, type=str, required=False, help="name for the output dipoles (default: 'model_dipole')", default="model_dipole")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file with the atomic structures and the predicted properties")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: 'None')", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading model from file '{:s}' ... ".format(args.model), end="")
    model = DipoleModel.from_file(args.model)
    print("done")
    #------------------#
    print("\n\tmodel summary: ")
    model.summary(string="\t\t")

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(structures)))

    #------------------#
    print("\n\tEvaluating the dipole model ... ", end="")
    output:np.ndarray = model.get(structures)
    print("done")

    #------------------#
    print("\n\tSetting the dipoles computed by the model as '{:s}' info ... ".format(args.name), end="")
    structures.set_info(name=args.name,data=output)
    print("done")

    #------------------#
    print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output,format=args.output_format)
    print("done")
     

#---------------------------------------#
if __name__ == "__main__":
    main()