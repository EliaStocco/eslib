#!/usr/bin/env python
import json
import os
import pandas as pd
from eslib.formatting import esfmt

#---------------------------------------#
description = "Returns the file path of the torch parameters with the lowest validation loss."

#---------------------------------------#
def prepare_args(description):
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-t" ,"--training"     ,  **argv, type=str  , help="training input file (default: 'input.json')", default="input.json")
    parser.add_argument("-i" ,"--instructions" ,  **argv, type=str  , help="model input file (default: 'instructions.json')", default="instructions.json")
    parser.add_argument("-f" ,"--folder"       ,  **argv, type=str  , help="folder where the previous files are stored (default: '.')", default=".")
    parser.add_argument("-bs","--batch_size"   ,  **argv, type=int  , help="batch size")
    parser.add_argument("-lr","--learning_rate",  **argv, type=float, help="learning rate")
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    file = os.path.normpath("{:s}/{:s}".format(args.folder,args.training))
    with open(file, 'r') as f:
        parameters = json.load(f)

    #------------------#
    # find bets parameters
    tmp = args.folder, parameters["output_folder"], parameters["name"], args.batch_size, args.learning_rate
    file = "{:s}/{:s}/dataframes/{:s}.bs={:d}.lr={:.1e}.csv".format(*tmp)
    if not os.path.exists(file):
        raise ValueError("file '{:s}' does not exist".format(file))
    loss = pd.read_csv(file)

    epoch = loss["val"].argmin()
    minloss = loss["val"].min()

    #------------------#
    tmp = args.folder, parameters["output_folder"], parameters["name"], args.batch_size, args.learning_rate
    par_folder = os.path.normpath("{:s}/{:s}/parameters/{:s}.bs={:d}.lr={:.1e}".format(*tmp))
    par_files = os.listdir(par_folder)
    best_parameters = None
    best_epoch = 0
    for file in par_files:
        #if parameters["name"] in file :
        tmp = int(file.split("epoch=")[1].split(".")[0])
        if tmp > best_epoch and tmp <= epoch :
            best_parameters = file
            best_epoch = tmp

    #------------------#
    best_parameters = os.path.normpath("{:s}/{:s}".format(par_folder,best_parameters))
    print("\n\tlowest loss:")
    print("\t{:>10s}: {:e}".format("value",minloss))
    print("\t{:>10s}: {:d}".format("epoch",epoch))
    print("\t{:>10s}: {:s}".format("file",best_parameters))

#---------------------------------------#
if __name__ == "__main__":
    main()