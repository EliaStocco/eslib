#!/usr/bin/env python
import argparse
# import json5 as json
import json
import os
import torch
import pandas as pd
import numpy as np
# from scipy.stats import pearsonr
# from copy import deepcopy
import time
import matplotlib.pyplot as plt
from elia.nn.functions.functions import get_model
# from elia.functions import plot_bisector
from elia.nn.dataset import make_dataloader

#####################

description = "Recompute the train and validation losses in 'eval' mode"

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    # Argument for "input"
    parser.add_argument(
        "--training", action="store", type=str,
        help="training input file", default="input.json"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--instructions", action="store", type=str,
        help="model input file", default="instructions.json"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--bs", action="store", nargs='+', type=int,
        help="batch size"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--lr", action="store", nargs='+', type=float,
        help="learning rate"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--max_time", action="store", type=int, default=-1,
        help="max_time"
    )


    return parser.parse_args()

def main():

    # Record the start time
    start_time = time.time()
    
    #####################
    # read input
    args = get_args()

    with open(args.training, 'r') as file:
        parameters = json.load(file)
        if "trial" not in parameters:
            parameters["trial"] = None
        elif parameters["trial"] in ["None",0,1]:
            parameters["trial"] = None      

    #####################
    if args.bs is None :
        args.bs = parameters["bs"]
    if args.lr is None :
        args.lr = parameters["lr"]

    #####################
    # get train dataset
    print("\n\tLoading datasets:")
    datasets = { "train":None, "val":None, "test":None }
    dataloader = { "train":None, "val":None, "test":None }

    for name in ["train","val"]:
        file = "{:s}/dataset.{:s}.torch".format(parameters["folder"],name)
        print("\t\t{:s}: {:s}".format(name,file))
        datasets[name] = torch.load(file)
        dataloader[name] = make_dataloader(dataset=datasets[name],batch_size=-1,shuffle=False,drop_last=False)

    #####################
    # evaluate train dataset
    if parameters["output"] == "D":
        real = {
            "train" : torch.zeros((len(datasets["train"]),3)),
            # "test"  : np.zeros((len(datasets["test"]),3)),
            "val"   : torch.zeros((len(datasets["val"]),3)),
        }            
         #ncomp = 3
        # colors = ["navy","red","green"]
        # labels = ["$P_x$","$P_y$","$P_z$"]
        get_real = lambda x : x.dipole
    elif parameters["output"] == "E":
        real = {
            "train" : torch.zeros((len(datasets["train"]),1)),
            # "test"  : np.zeros((len(datasets["test"]),1)),
            "val"   : torch.zeros((len(datasets["val"]),1)),
        }            
        # colors = ["navy"]
        # labels = ["energy"]
        # ncomp = 1
        get_real = lambda x : x.energy
    else :
        raise ValueError("not implemented yet")

    #####################
    print("\tGetting the real values of the datasets ...")
    pp_folder = "post-processing"
    if not os.path.exists(pp_folder):
        os.mkdir(pp_folder)

    # pred = deepcopy(real)

    for name in ["train","val"]:

        real_file = os.path.normpath("{:s}/real.{:s}.pth".format(pp_folder,name))

        if not os.path.exists(real_file):
            X = next(iter(dataloader[name]))
            N = len(datasets[name])
            y = get_real(X)
            y = y.detach() #.numpy()
            y = y.reshape((N,-1))
            real[name] = y
            torch.save(real[name],real_file)
        else :
            real[name] = torch.load(real_file)


    #####################
    tot_bs_lr = len(args.bs) * len(args.lr)
    model = None 
    for bs in args.bs :
        for lr in args.lr :

            max_time_cycle = args.max_time / tot_bs_lr

            print("bs:{:d} | lr={:.1e}".format(bs,lr))

            tmp = parameters["output_folder"], parameters["name"], bs, lr

            loss_file = "{:s}/dataframes/{:s}.bs={:d}.lr={:.1e}.csv".format(*tmp)
            old_loss = pd.read_csv(loss_file)

            par_folder = "{:s}/parameters/{:s}.bs={:d}.lr={:.1e}".format(*tmp)
            files = os.listdir(par_folder)

            loss = pd.DataFrame(columns=["epoch","train","val","old-train","old-val","train-2"]) 
            #,index=np.arange(len(files)))
            loss["old-train"] = old_loss["train"]
            loss["old-val"] = old_loss["val"]
            loss["epoch"] = np.arange(len(loss))

            ofile = "{:s}/{:s}.bs={:d}.lr={:.1e}.csv".format(pp_folder,parameters["name"], bs, lr)

            k = 0 
            for n,file in enumerate(files):

                elapsed_time = time.time() - start_time

                if elapsed_time > max_time_cycle - 10 and args.max_time != -1 :
                    break

                
                file = os.path.join(par_folder,file)
            
                epoch = int(file.split("epoch=")[1].split(".")[0])
                # loss.at[epoch,"epoch"] = epoch

                print("epoch:{:d}".format(epoch),end="")
                print(" | file: {:s}".format(file))
                
                if model is None :
                    model = get_model(args.instructions,file)
                    loss_fn = model.loss(Natoms=parameters["Natoms"])
                else :
                    checkpoint = torch.load(file)
                    model.load_state_dict(checkpoint)
                    model.eval()

                for name in ["train","val"]:

                    # pred_file = os.path.normpath("{:s}/real.{:s}.txt".format(pp_folder,name))

                    # if not os.path.exists(pred_file):
                    X = next(iter(dataloader[name]))                        
                    # del X.edge_vec
                    # real[name][n,:] = get_real(X).detach().numpy()
                    # pred[name][n,:] = model(X).detach().numpy()

                    N = len(datasets[name])
                    y = model(X)
                    y = y.detach() # .numpy()
                    y = y.reshape((N,-1))
                    # pred[name] = y

                    if name == "train" :
                        loss.at[epoch,"train-2"] = float(loss_fn(real[name],y))
                    elif name == "val" :
                        loss.at[epoch,"val"] = float(loss_fn(real[name],y))
                        # np.savetxt(pred_file,real[name])

                    # else :
                    #     pred[name] = np.loadtxt(pred_file)

                k += 1 

            loss.to_csv(ofile,index=False)

    print("\n\tJob done :)")

#####################

if __name__ == "__main__":
    main()