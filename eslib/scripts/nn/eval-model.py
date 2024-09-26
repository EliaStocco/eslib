#!/usr/bin/env python
import json
import numpy as np
import torch
from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models import eslibModel
from eslib.formatting import esfmt, float_format, warning
from eslib.input import slist, nilist, literal
from eslib.classes.physical_tensor import PhysicalTensor
import pandas as pd
from eslib.show import show_dict
from eslib.tools import is_integer

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
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=False, help="output file with the atomic structures and the predicted properties (default: %(default)s)", default="mace.extxyz")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    # Save data to txt/npy files
    parser.add_argument("-n" , "--names"        , **argv, type=literal, required=False, help="names for the info/arrays to be saved to txt/npy files (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--shapes"       , **argv, type=nilist         , required=False, help="data reshapes (default: %(default)s", default=None)  
    parser.add_argument("-do", "--data_output"  , **argv, type=literal, required=False, help="data output files (default: %(default)s)", default=None)
    parser.add_argument("-df", "--data_format"  , **argv, type=literal, required=False, help="output format for np.savetxt (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

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
            if not is_integer(c):
                print("\t{:s}: '{:s}' charge is not an integer".format(warning,k))
            charges[k] = np.round(c,0)
            
            
        #------------------#
        print("\n\tCreating dipole model based on the charges ... ",end="")
        model = DipolePartialCharges(charges)
        print("done")

        #------------------#
        print("\n\tAdding charges as '{:s}' to the 'arrays' of the atomic structures ... ".format(args.name),end="")
        for n,structure in enumerate(structures):
            if not model.check_charge_neutrality(structure):
                raise ValueError("structure . {:d} is not charge neutral".format(n))
            structure.arrays[args.charges] = model.get_all_charges(structure)
        print("done")        
    
    #------------------#
    if hasattr(model,"charges_key"):
        if not structures.is_there(model.charges_key):
            raise ValueError("The atomic structures do not have the key '{:s}'".format(model.charges_key))

    #------------------#
    print("\n\tEvaluating the MACE model ... ", end="")
    output:AtomicStructures = model.compute(structures,args.prefix)
    print("done")

    #------------------#
    print("\n\tSummary of the output atomic structures: ")
    df = output.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    

    #------------------#
    print("\n\tSaving atomic structures to file '{:s}' ... ".format(args.output), end="")
    output.to_file(file=args.output,format=args.output_format)
    print("done")
    
    #------------------#
    print("\n\tSaving info/arrays to file:")
    # args.names = slist(args.names)
    if args.names is not None and len(args.names) > 0 :
        if isinstance(args.names, str):
            args.names = [args.names]             
            args.shapes      = [args.shapes]      if args.shapes       is not None else [None]
            args.data_output = [args.data_output] if args.data_output is not None else [None]
            args.data_format = [args.data_format] if args.data_format is not None else [None]
        elif isinstance(args.names, list) or isinstance(args.names, np.ndarray):
            args.shapes      = args.shapes      if args.shapes       is not None else [None]*len(args.names)
            args.data_output = args.data_output if args.data_output is not None else [None]*len(args.names)
            args.data_format = args.data_format if args.data_format is not None else [None]*len(args.names)
        else:
            raise TypeError("args.names must be either a string or a list of strings, but got '{:s}'".format(type(args.names)))
        
        # args.data_output = slist(args.data_output)
        # args.data_format = slist(args.data_format)
        
        df = pd.DataFrame(columns=["name","shape","file","format"])
        # try:
        for name,shape,file,data_format in zip(args.names,args.shapes,args.data_output,args.data_format):
            print("\t - '{:s}' ... ".format(name), end="")
            data = output.get(name)
            if np.issubdtype(data.dtype, np.str_):
                data_format = "%s"
            elif data_format is None:
                data_format = "%r"
                data_format = float_format
            if shape is not None:
                data = data.reshape(shape)
            data = PhysicalTensor(data)
            data.to_file(file=file,fmt=data_format)
            print("done")
            df = pd.concat([df,\
                    pd.DataFrame([{"name":name,"shape":str(data.shape),"file":file,"format":str(data_format)}])],\
                    ignore_index=True)
        # except Exception as e:
        #    print(e)
        
    print("\n\tSummary of the info/arrays saved to file:")
    df = "\n"+df.to_string(index=False)
    print(df.replace("\n", "\n\t"))
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()