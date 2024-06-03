#!/usr/bin/env python
from ase.io import write
from eslib.classes.dipole import DipoleModel
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Extract the reference structure of a dipole linear model."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"          , **argv, type=str, help="pickle file with the dipole linear model (default: 'DipoleModel.pickle')", default='DipoleModel.pickle')
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file with the reference structure (default: 'reference.extxyz')", default="reference.extxyz")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # linear model
    print("\tLoading the dipole linear model from file '{:s}' ... ".format(args.input), end="")
    model = DipoleModel.from_pickle(args.input)
    print("done")

    #------------------#
    # atomic structure
    print("\tExtracting the reference structure ... ", end="")
    reference = model.get_reference()
    print("done")
    
       
    print("\n\tWriting reference structure to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=reference,filename=args.output) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    

#---------------------------------------#
if __name__ == "__main__":
    main()