#!/usr/bin/env python
import json

import numpy as np
import torch

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models import eslibModel
from eslib.formatting import esfmt, warning, float_format
from eslib.show import show_dict
from eslib.tools import is_integer
from eslib.io_tools import read_json

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate a model."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # Input
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--model"        , **argv, type=str, required=True , help="*.pth file with the MACE model of JSON file with instructions")
    parser.add_argument("-c" , "--charges"      , **argv, required=False, type=str, help="charges name (default: %(default)s)", default=None)
    parser.add_argument("-cf", "--charges_file" , **argv, required=False, type=str, help="charges file (default: %(default)s)", default=None)
    parser.add_argument("-p" , "--prefix"       , **argv, type=str, required=False, help="prefix to be prepended to the properties evaluated by the MACE model (default: %(default)s)", default="MACE_")
    parser.add_argument("-o" , "--structures"       , **argv, type=str, required=False, help="structures file with the atomic structures and the predicted properties (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="structures file format (default: %(default)s)", default=None)
    # Save data to txt/npy files
    parser.add_argument("-oia" , "--output_info_array", **argv, type=str, required=False, help="JSON file with the instructions to save info and array t file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.structures is None:
        print(f"\n\t{warning}: no structures file will be printed.")
    if args.output_info_array is None:
        print(f"\n\t{warning}: no info or arrays will be saved to file.")
    else:
        print("\n\tReading instructions from file '{:s}' ... ".format(args.output_info_array), end="")
        instructions = read_json(args.output_info_array)
        print("done")
        
        for key, instr in instructions.items():
            assert "file" in instr, f"No 'file' key in instructions for '{key}'."
            
    #------------------#
    print("\tCuda available: ",torch.cuda.is_available())
                 
    #------------------#
    if str(args.model).endswith(".json"):
        print("\tReading MACE model input parameters from file '{:s}' ... ".format(args.model), end="")
        with open(args.model, 'r') as json_file:
            kwargs = json.load(json_file)
        print("done")

        print("\tAllocating the model ... ", end="")
        model = eslibModel(**kwargs)
        print("done")
    else:
        print("\tLoading model from file '{:s}' ... ".format(args.model), end="")
        model = eslibModel.from_file(args.model)
        print("done")

    #------------------#
    if args.charges is not None:
        print("\n\tReplacing charges key: '{:s}' --> '{:s}'".format(model.charges_key,args.charges))
        model.charges_key = args.charges

    #------------------#
    model.summary()

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(structures)))
    
    #------------------#
    if args.charges_file is not None:
        
        from eslib.classes.models.dipole import DipolePartialCharges

        #------------------#
        # charges
        print("\tReading the charges from file '{:s}' ... ".format(args.charges_file), end="")
        with open(args.charges_file, 'r') as json_file:
            charges:dict = json.load(json_file)
        print("done")
        
        #------------------#
        print("\n\tCharges: ")
        show_dict(charges,"\t\t",2)
        
        for k,c in charges.items():
            assert is_integer(c), f"'{k}' charge is not an integer."                  
            
        #------------------#
        print("\n\tCreating dipole model based on the charges ... ",end="")
        charges_model = DipolePartialCharges(charges=charges)
        print("done")

        #------------------#
        print("\n\tAdding charges as '{:s}' to the 'arrays' of the atomic structures ... ".format(args.charges),end="")
        for n,structure in enumerate(structures):
            if not charges_model.check_charge_neutrality(structure):
                raise ValueError("structure . {:d} is not charge neutral".format(n))
            structure.arrays[args.charges] = charges_model.get_all_charges(structure)
        print("done")        
    
    #------------------#
    if hasattr(model,"charges_key"):
        if not structures.is_there(model.charges_key):
            raise ValueError("The atomic structures do not have the key '{:s}'".format(model.charges_key))

    #------------------#
    print("\n\tEvaluating the MACE model ... ", end="")
    # overwrite structures to save space in memory
    structures:AtomicStructures = model.compute(structures,args.prefix)
    print("done")

    #------------------#
    print("\n\tSummary of the structures atomic structures: ")
    df = structures.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    
    #------------------#
    if args.structures is not None:
        print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.structures), end="")
        structures.to_file(file=args.structures,format=args.output_format)
        print("done")
    
    #------------------#
    if args.output_info_array is not None:
        print("\n\tReading instructions from file '{:s}' ... ".format(args.output_info_array), end="")
        instructions = read_json(args.output_info_array)
        print("done")
        
        print("\n\tSaving info and array to file:", end="")
        for key, instr in instructions.items():
            print(f"\n\t - '{key}':", end="")
            array = structures.get(key)
            print(f"\n\t\t extracted with shape {array.shape}:", end="")
            if "shape" in instr:
                array = np.reshape(array, instr["shape"])
                print(f"\n\t\t reshaped to {array.shape}:", end="")
                file = str(instr["file"])
                print(f"\n\t\t saved to {file}:", end="")
                if file.endswith("txt"):
                    np.savetxt(file,array,fmt=float_format)
                elif file.endswith("npy"):
                    np.save(file,array)
                else:
                    raise ValueError("Only `txt` and `npy` extensions are supported.")
        print()
    
#---------------------------------------#
if __name__ == "__main__":
    main()