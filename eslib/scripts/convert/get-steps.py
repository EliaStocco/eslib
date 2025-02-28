#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt
from eslib.io_tools import read_comments_extxyz

#---------------------------------------#
# Description of the script's purpose
description = "Split an (ext)xyz file."

#---------------------------------------#
def prepare_args(description):
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , required=True ,**argv,type=str, help="extxyz input file")
    parser.add_argument("-o" , "--output"       , required=True ,**argv,type=str, help="output txt file")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print("\n\tReading comments from file '{:s}' ... ".format(args.input),end="")
    comments = read_comments_extxyz(args.input)
    print("done")
    print("\tNumber of comment lines: {:d}".format(len(comments)))
    
    #------------------#
    print("\n\tExtracting steps from comments ... ",end="")
    indices = np.zeros(len(comments),dtype=int)
    for i,comment in enumerate(comments):
        step = int(comment.split("Step:")[1].split()[0])
        indices[i] = step
    print("done")
    
    #------------------#
    print("\n\tWriting steps to file '{:s}' ... ".format(args.output),end="")
    np.savetxt(args.output, indices, fmt="%d")
    print("done")            
    
#---------------------------------------#
if __name__ == "__main__":
    main()
