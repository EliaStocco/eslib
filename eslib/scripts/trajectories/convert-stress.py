#!/usr/bin/env python
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

#---------------------------------------#
# Description of the script's purpose
description = "Convert stress to or from Voigt notation."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , type=str, **argv, required=True , help='input extxyz file')
    parser.add_argument("-ik", "--input_keyword" , type=str, **argv, required=True , help="input stress keyword")
    parser.add_argument("-ok", "--output_keyword", type=str, **argv, required=True , help="output stress keyword")
    parser.add_argument("-s" , "--shape"         , type=str, **argv, required=True , help="output shape (e.g. 'voigt' for (6,), 'cartesian' for 3x3 tensor)", default='voigt', choices=['voigt','cartesian'])   
    parser.add_argument("-o" , "--output"        , type=str, **argv, required=True, help="output file")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structures
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    structures = AtomicStructures.from_file(file=args.input)
    print("done")

    #------------------#
    print(f"\tExtracting stress tensors from '{args.input_keyword}' from the trajectory ... ", end="")
    stress = structures.get(args.input_keyword,np.full((6,),np.nan),"info")  
    print("done")
    try:
        print(f"\t'{args.input_keyword}' shape: ",stress.shape)
    except:
        pass
    
    #------------------#
    if args.shape == "voigt":
        stress = np.asarray([
            full_3x3_to_voigt_6_stress(s) if s.shape == (3, 3) else s
            for s in stress
        ])
    else:
        stress = np.asarray([
            voigt_6_to_full_3x3_stress(s) if s.shape == (6,) else s
            for s in stress
        ])
       
    #------------------# 
    print(f"\tSetting the new stress tensors to '{args.output_keyword}' ... ", end="")
    structures.set(args.output_keyword,stress,"info")  
    print("done")
    
    #------------------#
    print(f"\tExtracting stress tensors from '{args.output_keyword}' from the trajectory ... ", end="")
    stress = structures.get(args.output_keyword,None,"info")  
    print("done")
    print(f"\t'{args.output_keyword}' shape: ",stress.shape)
    
    #---------------------------------------#
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")

    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
