#!/usr/bin/env python
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Format BEC in a extxyz file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , type=str, **argv, required=True , help="atomic structures file [extxyz]")
    parser.add_argument("-if", "--input_format"  , type=str, **argv, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-ik", "--input_keyword" , type=str, **argv, required=True , help="input BEC keyword")
    parser.add_argument("-ok", "--output_keyword", type=str, **argv, required=True , help="output BEC keyword")
    parser.add_argument("-w" , "--what"          , type=str, **argv, required=True , help="what to do ('a': aggregate, 'd': distribute)", choices=["a","d"])
    parser.add_argument("-a" , "--axis"          , type=int, **argv, required=True , help="axis corresponding to the dipole component",choices=[1,2])
    parser.add_argument("-o" , "--output"        , type=str, **argv, required=True , help="output file")
    parser.add_argument("-of" , "--output_format", type=str, **argv, required=False, help="output file format (default: %(default)s)", default=None)
    return parser
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print(f"\tReading atomic structures from file '{args.input}' ... ", end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    
    #------------------#
    Natoms = structures.call(lambda x: x.get_global_number_of_atoms())
    if args.what == "a": # aggregate
        BECx, BECy, BECz  = [ structures.get(f"{args.input_keyword}{a}",None,"arrays") for a in ["x","y","z"]]
        BEC = [None]*len(structures)
        for n,Na in enumerate(Natoms):
            BEC[n] = np.zeros((Na,3,3))
    
            if args.axis == 1:
                BEC[n][:,0,:] = BECx[n]
                BEC[n][:,1,:] = BECy[n]
                BEC[n][:,2,:] = BECz[n]
            elif args.axis == 2:
                BEC[n][:,:,0] = BECx[n]
                BEC[n][:,:,1] = BECy[n]
                BEC[n][:,:,2] = BECz[n]
                
            BEC[n] = BEC[n].reshape((Na,9))
        
        print(f"\tSetting the new BEC tensors to '{args.output_keyword}' ... ", end="")
        structures.set(args.output_keyword,BEC,"arrays")
        print("done")
                
    else: # distribute
        BEC = structures.get(args.input_keyword,None,"arrays")
        BECx, BECy, BECz  = [ [None]*len(structures) for _ in range(3)]
        for n,Na in enumerate(Natoms):
            if args.axis == 1:
                BECx[n] = BEC[n].reshape((Na,3,3))[:,0,:]
                BECy[n] = BEC[n].reshape((Na,3,3))[:,1,:]
                BECz[n] = BEC[n].reshape((Na,3,3))[:,2,:]
            elif args.axis == 2:
                BECx[n] = BEC[n].reshape((Na,3,3))[:,:,0]
                BECy[n] = BEC[n].reshape((Na,3,3))[:,:,1]
                BECz[n] = BEC[n].reshape((Na,3,3))[:,:,2]
        
        for z,a in zip([BECx,BECy,BECz],["x","y","z"]):
            print(f"\tSetting the new BEC{a} tensors to '{args.output_keyword}{a}' ... ", end="")
            structures.set(f"{args.input_keyword}{a}",z,"arrays")
            print("done")
        
    #------------------#
    print(f"\n\tWriting data to file '{args.output}' ... ", end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")
    
    return    
    
#---------------------------------------#
if __name__ == "__main__":
    main()