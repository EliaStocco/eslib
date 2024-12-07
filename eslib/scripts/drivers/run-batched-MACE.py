#!/usr/bin/env python
import torch
from eslib.formatting import esfmt
from eslib.input import slist
from eslib.classes.calculators.fileIObatchMACE import FileIOBatchedMACE

#---------------------------------------#
# Description of the script's purpose
description = "Run a many FileIOBatchedMACE."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-f" , "--folders" , **argv, required=True , type=slist, help="list of folders")
    parser.add_argument("-m" , "--model"   , **argv, required=False, type=str  , help="file with the MACE model (default: %(default)s)", default=None)
    parser.add_argument("-l" , "--log_file", **argv, required=False, type=str  , help="optional path for the log file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tCuda available: ",torch.cuda.is_available())
    
    #------------------#
    print("\tAllocating the calculator ... ", end="")
    machine = FileIOBatchedMACE(folders=args.folders,model=args.model,log_file=args.log_file)
    print("done")
    
    #------------------#
    print("\n\tRunning ... ", end="")
    machine.run()
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()