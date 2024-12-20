#!/usr/bin/env python
import json
import os 

import numpy as np
import torch

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models import eslibModel
from eslib.formatting import esfmt, warning, float_format, eslog
from eslib.show import show_dict
from eslib.tools import is_integer
from eslib.io_tools import read_json, add_index2file, pattern2sorted_files
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate a model for many files."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # Input
    parser.add_argument("-i" , "--input"        , **argv, type=str     , required=True , help="list of files with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str     , required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--model"        , **argv, type=str     , required=True , help="*.pth file with the MACE model of JSON file with instructions")
    parser.add_argument("-c" , "--charges"      , **argv, type=str     , required=False, help="charges name (default: %(default)s)", default=None)
    parser.add_argument("-cf", "--charges_file" , **argv, type=str     , required=False, help="charges file (default: %(default)s)", default=None)
    parser.add_argument("-p" , "--prefix"       , **argv, type=str     , required=False, help="prefix to be prepended to the properties evaluated by the MACE model (default: %(default)s)", default="MACE_")
    parser.add_argument("-o" , "--output"       , **argv, type=str     , required=False, help="output file with the atomic structures and the predicted properties (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format", **argv, type=str     , required=False, help="structures file format (default: %(default)s)", default=None)
    parser.add_argument("-ow", "--over_write"   , **argv, type=str2bool, required=False, help="whether to over write the output file (default: %(default)s)", default=True)
    # Save data to txt/npy files
    parser.add_argument("-oia" , "--output_info_array", **argv, type=str, required=False, help="JSON file with the instructions to save info and array t file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.output is None:
        print(f"\n\t {warning}: no structures file will be printed.")
    if args.output_info_array is None:
        print(f"\n\t {warning}: no info or arrays will be saved to file.")
    else:
        print("\n\t Reading instructions from file '{:s}' ... ".format(args.output_info_array), end="")
        instructions = read_json(args.output_info_array)
        print("done")
        
        for key, instr in instructions.items():
            assert "file" in instr, f"No 'file' key in instructions for '{key}'."
                
    #------------------#
    print("\t Cuda available: ",torch.cuda.is_available())
                 
    #------------------#
    if str(args.model).endswith(".json"):
        print("\t Reading MACE model input parameters from file '{:s}' ... ".format(args.model), end="")
        with open(args.model, 'r') as json_file:
            kwargs = json.load(json_file)
        print("done")

        print("\t Allocating the model ... ", end="")
        model = eslibModel(**kwargs)
        print("done")
    else:
        print("\t Loading model from file '{:s}' ... ".format(args.model), end="")
        model = eslibModel.from_file(args.model)
        print("done")

    #------------------#
    if args.charges is not None:
        print("\n\t Replacing charges key: '{:s}' --> '{:s}'".format(model.charges_key,args.charges))
        model.charges_key = args.charges

    #------------------#
    model.summary()
    
    #------------------#
    if args.charges_file is not None:
        
        from eslib.classes.models.dipole import DipolePartialCharges

        #------------------#
        # charges
        print("\t Reading the charges from file '{:s}' ... ".format(args.charges_file), end="")
        with open(args.charges_file, 'r') as json_file:
            charges:dict = json.load(json_file)
        print("done")
        
        #------------------#
        print("\n\t Charges: ")
        show_dict(charges,"\t\t",2)
        
        for k,c in charges.items():
            assert is_integer(c), f"'{k}' charge is not an integer."                  
            
        #------------------#
        print("\n\t Creating dipole model based on the charges ... ",end="")
        charges_model = DipolePartialCharges(charges=charges)
        print("done")

    all_files = pattern2sorted_files(args.input)
    for n_file,file in enumerate(all_files):
        
        if not args.over_write:
            shall_compute = False
            if args.output is not None:
                new_output = add_index2file(args.output,n_file)
                shall_compute = shall_compute or not os.path.exists(new_output)
                    
            if args.output_info_array is not None:
                for key, instr in instructions.items():
                    file = add_index2file(str(instr["file"]),n_file)
                    shall_compute = shall_compute or not os.path.exists(file)
                    
            if not shall_compute:
                print("\t Skipping '{:s}' ... ".format(file))
                continue
        
        print("\n\t#---------------------------------------#\n")
        
        #------------------#
        # trajectory
        with eslog(f"Reading the atomic structures from file '{file}'"):
            structures = AtomicStructures.from_file(file=file,format=args.input_format)
        print("\t n. of structures: {:d}".format(len(structures)))
    
        #------------------#
        if args.charges_file is not None:

            print("\n\t Adding charges as '{:s}' to the 'arrays' of the atomic structures ... ".format(args.charges),end="")
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
        with eslog(f"\nEvaluating the MACE model"):
            # overwrite structures to save space in memory
            structures:AtomicStructures = model.compute(structures,args.prefix)
    
        #------------------#
        print("\n\t Summary of the structures atomic structures: ")
        df = structures.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t"))
        
        #------------------#
        if args.output is not None:
            print("\n\t Saving atomic structures to file '{:s}' ... ".format(new_output), end="")
            structures.to_file(new_output,format=args.output_format)
            print("done")
    
        #------------------#
        if args.output_info_array is not None:
            
            print("\n\t Saving info and array to file:")
            for key, instr in instructions.items():
                print(f"\t - '{key}':")
                array = structures.get(key)
                print(f"\t\t extracted with shape {array.shape}:")
                if "shape" in instr:
                    array = np.reshape(array, instr["shape"])
                    print(f"\t\t reshaped to {array.shape}:")
                    file = add_index2file(str(instr["file"]),n_file)
                    print(f"\t\t saved to {file}:")
                    if file.endswith("txt"):
                        np.savetxt(file,array,fmt=float_format)
                    elif file.endswith("npy"):
                        np.save(file,array)
                    else:
                        raise ValueError("Only `txt` and `npy` extensions are supported.")
        
    print("\n\t#---------------------------------------#\n")
        
#---------------------------------------#
if __name__ == "__main__":
    main()