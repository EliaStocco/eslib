#!/usr/bin/env python
import time
start_time = time.time()
import numpy as np
import random
import json
from copy import copy
import torch
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
from eslib.nn.training import hyper_train_at_fixed_model
# from elia.nn.network import aile3nn
from eslib.functions import add_default, str2bool
from eslib.nn.user import get_class
from eslib.formatting import esfmt
from warnings import warn

#---------------------------------------#
# Documentation
# - https://pytorch.org/docs/stablfe/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#---------------------------------------#
description = "train a 'e3nn' model"

#---------------------------------------#
default_values = {
        # "class"            : "aile3nn",
        # "module"           : "elia.nn.network",
        # "mul"              : 2,
        # "layers"           : 6,
        # "lmax"             : 2,
        "name"             : "untitled",
        # "output"           : "D",
        # "max_radius"       : 6.0,
        # "datasets"         : {
        #     "train" : "data/dataset.train.pth",
        #     "val"   : "data/dataset.val.pth",
        # },
        "output_folder"     : "output",
        "checkpoint_folder" : "checkpoints",
        "info-file"         : "info.csv",
        # "Natoms"           : None,
        "random"            : False,
        "epochs"            : 10,
        "bs"                : [1],
        "lr"                : [1e-3],
        "weight_decay"      : 1e-2,
        "optimizer"         : "adam",
        "grid"              : True,
        "max_time"          : -1,
        "task_time"         : -1,
        "dropout"           : 0.01,
        # "batchnorm"        : True,
        # "use_shift"        : None,
        "restart"           : True,
        "recompute_loss"    : False,
        # "fixed_charges_only": False,
        # "instructions"     : None,
        "debug"             : False,
        # "indices"          : None,
        "options"           : None,
        "scheduler"         : "",
        "scheduler-factor"  : 1e-2,
        "init-parameters"   : None,
    }

#---------------------------------------#
def get_args(description):
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("-i", "--input", type=str, help="json file with the input parameters")
    parser.add_argument("-n", "--network" , type=str, required=True, help="JSON file with the parameters to allocate the network")
    parser.add_argument("-t", "--training", type=str, required=True, help="JSON file with the parameters for the training")
    parser.add_argument("-d", "--datasets", type=str, required=True, help="JSON file with the path to the train and validation datasets (*.pth files)")
    return parser.parse_args()

#---------------------------------------#
def check_parameters(parameters):
    
    str2bool_keys = ["random","grid","recompute_loss","debug"]
    for k in str2bool_keys : 
        parameters[k] = str2bool(parameters[k])
    
    if parameters["task_time"] <= 0 :
        parameters["task_time"] = -1

    if parameters["max_time"] <= 0 :
        parameters["max_time"] = -1

    # if "chemical-species" not in parameters \
    #     or parameters["chemical-species"] is None \
    #         or len(parameters["chemical-species"]) == 0 :
    #     raise ValueError("Please specify a list of the chemical species that compose the provided atomic structures.")

#---------------------------------------#
def get_parameters(args):
    """Returns the training parameters and the parameters to allocate the network."""

    #------------------#
    parameters = None
    if args.training is not None :
        try :
            with open(args.training, 'r') as file:
                parameters = json.load(file)
        except :
            raise ValueError(f'error reading \'{args.training}\' file')
        parameters = add_default(parameters,default_values)
    else :
        raise ValueError("no file for training parameters provided (--training)")
    check_parameters(parameters)

    # print parameters
    print("\n\tTraining parameters:")
    for k in parameters.keys():
        print("\t\t{:20s}: ".format(k),parameters[k])

    #------------------#
    network = None
    if args.network is not None :
        try :
            with open(args.network, 'r') as file:
                network = json.load(file)
        except :
            raise ValueError(f'error reading \'{args.network}\' file')
        # network = add_default(network,default_values)
    else :
        raise ValueError("no file for network parameters provided (--network)")

    # print("\n\tNetwork parameters:")
    # for k in network.keys():
    #     print("\t\t{:20s}: ".format(k),network[k])
    #------------------#
    if isinstance(args.datasets,str):
        with open(args.datasets, 'r') as file:
            datasets = json.load(file)
    elif isinstance(args.datasets,dict):
        datasets = args.datasets
    else:
        raise TypeError("'datasets' can be 'dict' or 'str' only.")

    #------------------#
    return network, parameters, datasets

#---------------------------------------#
# read datasets
def read_datasets(files:dict):
    dataset = dict()
    for k,file in files.items():
        dataset[k] = torch.load(file)
    return dataset

#---------------------------------------#
@esfmt(get_args,description)
def main(args):

    #------------------#
    # get user parameters
    network, parameters, datasets = get_parameters(args)    

    #------------------#
    if not parameters["random"] :
        # Set the seeds of the random numbers generators.
        # This is important for reproducitbility:
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
   
    # if "datasets" not in parameters:
    #     raise ValueError("please provide a dict in the input file to specify where to find the datasets.")
    # datasets = read_datasets(parameters["datasets"])
    datasets = read_datasets(datasets)

    #------------------#
    # test
    # # Let's do a simple test!
    # # If your NN is not working, let's focus only on one datapoint!
    # # The NN should train and the loss on the validation dataset get really high
    # # If this does not happen ... there is a bug somewhere
    # # You can also read this post: 
    # # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn

    if parameters["debug"] :
        print("\n\tModifying datasets for debugging")
        train_dataset = datasets["train"]
        val_dataset   = datasets["val"]
        
        if "n_debug" in parameters :
            train_dataset = train_dataset[0:parameters["n_debug"]["train"]] 
            val_dataset   = val_dataset  [0:parameters["n_debug"]["val"]] 
        else :
            train_dataset = train_dataset[0:1] 
            val_dataset   = val_dataset  [0:1] 

        print("\tDatasets summary:")
        print("\t\ttrain:",len(train_dataset))
        print("\t\t  val:",len(val_dataset))

        datasets = {
            "train":train_dataset,\
            "val"  :val_dataset
        }
        
        parameters["bs"] = [len(train_dataset)]

    ##########################################
    # construct the model
    # types = parameters["chemical-species"]
    # irreps_in = "{:d}x0e".format(len(types))

    # if parameters["output"] in ["E","EF"]:
    #     irreps_out = "1x0e"
    # elif parameters["output"] in ["ED","EDF"]:
    #     irreps_out = "1x0e + 1x1o"
    # elif parameters["output"] == "D":
    #     irreps_out = "1x1o"
    
    # #------------------#
    # kwargs = {
    #     "output"              : parameters["output"],
    #     "irreps_in"           : irreps_in,                  
    #     "irreps_out"          : irreps_out,                
    #     "max_radius"          : parameters["max_radius"],  
    #     "num_neighbors"       : 2,                      
    #     "pool_nodes"          : True,                      
    #     # "num_nodes"           : 2,
    #     "number_of_basis"     : 10,
    #     "mul"                 : parameters["mul"],
    #     "layers"              : parameters["layers"],
    #     "lmax"                : parameters["lmax"],
    #     "dropout_probability" : parameters["dropout"],
    #     # "pbc"                 : parameters["pbc"],
    #     "use_shift"           : parameters["use_shift"]
    # }

    #------------------#
    to_check = {"module","class","kwargs"}
    for s in to_check:
        if s not in network:
            raise ValueError(f'\'network\' does not have \'{s}\' key.')
    if set(network.keys()) != to_check:
        warn(f'keys in \'network\' different from \'module\', \'class\', and \'kwargs\' will be ignored.')
    
    cls = get_class(network["module"],network["class"])

    # if parameters["class"] == "aile3nnOxN":
    #     kwargs["fixed_charges_only"] = parameters["fixed_charges_only"]

    # instructions = {
    #         "kwargs"           : copy(kwargs),
    #         "class"            : parameters["class"],
    #         "module"           : parameters["module"],
    #     }
    
    # with open("instructions.json", "w") as json_file:
    #     json.dump(instructions, json_file, indent=4)

    net = cls(**network["kwargs"])
    try:
        N = net.n_parameters()
        print("Tot. number of parameters: ",N)
    except:
        pass

    #------------------#
    # initialize the parameters if a file is provided
    pfile = parameters["init-parameters"]
    if pfile is not None:
        print("Reading initial parameters from file '{:s}'".format(pfile))
        try:
            net.load_state_dict(torch.load(pfile),strict=False)
        except:
            print("Problems reading '{:s}'".format(pfile))

    #------------------#
    # choose the loss function
    # if parameters["output"] in ["D","E"] :
    loss = net.loss(Natoms=parameters["Natoms"] if "Natoms" in parameters else None)
    # elif parameters["output"] == "EF" :
    #     raise ValueError("not implemented yet")
    
    #------------------#
    # optional settings
    opts = {
            "plot":{
                "learning-curve" : {"N":10},
                "correlation" : {"N":-1}
            },
            "thr":{
                "exit":1e7
            },
            "save":{
                "parameters":1,
                "checkpoint":1,
            },
            "start_time"     : start_time,
            'keep_dataset'   : True,
            "restart"        : parameters["restart"],
            "recompute_loss" : parameters["recompute_loss"],
        }

    if parameters["options"] is not None :
        # read parameters from file
        try :
            with open(parameters["options"], 'r') as file:
                options = json.load(file)

        except :
            raise ValueError("error reading options file")
        # it should not be needed ...
        opts = add_default(options,opts)

    #------------------#
    # hyper-train the model
    hyper_train_at_fixed_model( net        = net,
                                all_bs     = parameters["bs"],
                                all_lr     = parameters["lr"],
                                epochs     = parameters["epochs"],
                                loss       = loss,
                                datasets   = datasets,
                                opts       = opts,
                                parameters = parameters
                            )

#---------------------------------------#
if __name__ == "__main__":
    main()
