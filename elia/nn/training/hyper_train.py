import numpy as np
import pandas as pd
from copy import copy
import torch
import os
from typing import Union
from .train import train
from elia.functions import add_default
from itertools import product

def hyper_train_at_fixed_model( net:torch.nn.Module,\
                                all_bs:list,\
                                all_lr:list,\
                                epochs,\
                                loss:Union[callable,str],\
                                datasets:dict,\
                                parameters:dict,\
                                opts:dict=None):
    
    ##########################################
    # set optional settings
    default = {"max_try" : 5}
    opts = add_default(opts,default)

    ##########################################
    # epochs
    if type(epochs) == int:
        rows = len(all_bs)
        cols = len(all_lr)
        data = np.full((rows, cols), epochs)
        epochs = pd.DataFrame(data, index=all_bs, columns=all_lr)

    elif type(epochs) == pd.DataFrame:
        pass
    else :
        raise ValueError("'epochs' type not supported")
    
    ##########################################
    # extract datasets
    train_dataset = datasets["train"]
    val_dataset   = datasets["val"]
    del datasets

    ##########################################
    # preparing cycle over hyper-parameters
    if parameters["grid"] :
        hyper_pars = list(product(all_bs, all_lr))
        Ntot = len(all_bs)*len(all_lr)
    else :
        if len(all_bs) != len(all_lr) :
            raise ValueError("batch sizes and learning rates should have the same lenght when 'grid'=True")
        hyper_pars = list(zip(all_bs, all_lr))
        Ntot = len(all_bs)

    #Ntot = len(hyper_pars)
    df = pd.DataFrame(columns=["bs","lr","file"],index=np.arange(Ntot))

    if parameters["task_time"] == -1 and parameters["max_time"] != 1 :
        parameters["task_time"] = ( parameters["max_time"] - 60 ) / Ntot

    
    ##########################################
    # looping over all hyper-parameters
    n = 0 
    init_model = copy(net) 
    for n in range(len(hyper_pars)) :
        bs,lr = hyper_pars[n]
        info = "all good"

        df.at[n,"bs"] = bs
        df.at[n,"lr"] = lr

        df.at[n,"file"] = "{:s}.bs={:d}.lr={:.1e}".format(parameters["name"],bs,lr)
        
        print("\n--------------------------------\n")
        print("\tbs={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(bs,lr,n+1,Ntot))

        #print("\n\trebuilding network...\n")
        net = copy(init_model)
        
        new_pars = add_default({
            "bs"       : bs,
            "n_epochs" : epochs.at[bs,lr],
            "name"     : df.at[n,"file"],
            "lr"       : lr,
            "loss"     : loss,
            "output"   : parameters["output_folder"],
        },parameters)

        count_try = 0
        while (info == "try again" and count_try < opts["max_try"]) or count_try == 0 :

            if info == "try again":
                print("\nLet's try again\n")

            model, arrays, info = \
                train(  model           = net,
                        train_dataset   = train_dataset,
                        val_dataset     = val_dataset,
                        parameters      = new_pars,
                        opts            = opts,
                    )
            count_try += 1

        if info == "try again":
            print("\nAborted training. Let's go on!\n") 

        df.at[n,"file"] = df.at[n,"file"] + ".pdf"
        n += 1

        df[:n].to_csv("temp-info.csv",index=False)

        n += 1
        if info == "exit file detected":
            break

    ##########################################
    # remove EXIT file if detected
    if os.path.exists("EXIT"):
        os.remove("EXIT")
        print("\n\t'exit' file removed\n")

    ##########################################
    # finish
    try : 
        # write information to file 'info.csv'
        df.to_csv("info.csv",index=False)

        # remove 'temp-info.csv'
        file_path = "temp-info.csv"  # Replace with the path to your file
        try:
            os.remove(file_path)
            print("File '{:s}' deleted successfully.".format(file_path))
        except OSError as e:
            print("Error deleting file '{:s}'".format(e))
    except OSError as e:
        print("Error writing file '{:s}'".format(e))

    pass