#!/usr/bin/env python
import numpy as np
from eslib.io_tools import pattern2sorted_files
from eslib.formatting import esfmt, float_format

#---------------------------------------#
# Description of the script's purpose
description = "Columns stack files."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input" , **argv, required=True , type=str, help="input files")
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output file")
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    files = pattern2sorted_files(args.input)
    print("done")
    print("\tn. of files: ",len(files))
    
    #------------------#
    # Construct the 'cat' command
    arrays = [None]*len(files)
    
    print("\n\tReading files:")
    for n,file in enumerate(files):
        print("\t - '{:s}' ... ".format(file), end="")
        arrays[n] = np.loadtxt(file)
        print("done")
        
    #------------------#
    print("\n\tStacking arrays ... ", end="")
    stacked_array:np.ndarray = np.column_stack(arrays)
    print("done")
    print("\tshape: ",stacked_array.shape)
    
    #------------------#
    print("\n\tWriting the stacked array into file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output,stacked_array,fmt=float_format)
    print("done")    

#---------------------------------------#
if __name__ == "__main__":
    main()