#!/usr/bin/env python
import argparse
# import json5 as json
import json
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
import matplotlib.pyplot as plt
from eslib.nn.functions.functions import get_model
from eslib.functions import plot_bisector

#####################

description = "post-process a 'e3nn' model training"

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-t","--training",  type=str,help="training input file (default: 'input.json')", default="input.json")
    parser.add_argument("-i","--instructions",  type=str,help="model input file (default: 'instructions.json')", default="instructions.json")
    parser.add_argument("-f","--folder",  type=str,help="folder (default: '.')", default=".")
    parser.add_argument("-bs","--batch_size",  type=int,help="batch size")
    parser.add_argument("-lr","--learning_rate",  type=float, help="learning rate")
    parser.add_argument("-b","--best_only",  type=bool,help="search the best file only (default=true)", default=True)
    return parser.parse_args()

def main():
    
    #####################
    # read input
    args = get_args()

    file = os.path.normpath("{:s}/{:s}".format(args.folder,args.training))
    with open(file, 'r') as f:
        parameters = json.load(f)

    #####################
    # find bets parameters
    tmp = args.folder, parameters["output_folder"], parameters["name"], args.batch_size, args.learning_rate
    file = "{:s}/{:s}/dataframes/{:s}.bs={:d}.lr={:.1e}.csv".format(*tmp)
    if not os.path.exists(file):
        raise ValueError("file '{:s}' does not exist".format(file))
    loss = pd.read_csv(file)

    epoch = loss["val"].argmin()

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

    best_parameters = os.path.normpath("{:s}/{:s}".format(par_folder,best_parameters))
    print("\n\tlowest loss file: {:s}\n".format(best_parameters))

    if args.best_only :
        print("\tJob done :)")
        return

    #####################
    # get model
    file = os.path.normpath("{:s}/{:s}".format(args.folder,args.instructions))
    model = get_model(file,best_parameters)

    #####################
    # get train dataset
    print("\n\tLoading datasets:")
    datasets = { "train":None, "val":None, "test":None }

    for name in ["train","val","test"]:
        file = os.path.normpath("{:s}/{:s}/dataset.{:s}.torch".format(args.folder,parameters["folder"],name))
        print("\t\t{:s}: {:s}".format(name,file))
        datasets[name] = torch.load(file)

    #####################
    # evaluate train dataset
    if parameters["output"] == "D":
        real = {
            "train" : np.zeros((len(datasets["train"]),3)),
            "test"  : np.zeros((len(datasets["test"]),3)),
            "val"   : np.zeros((len(datasets["val"]),3)),
        }            
        ncomp = 3
        colors = ["navy","red","green"]
        labels = ["$P_x$","$P_y$","$P_z$"]
        get_real = lambda x : x.dipole
    elif parameters["output"] == "E":
        real = {
            "train" : np.zeros((len(datasets["train"]),1)),
            "test"  : np.zeros((len(datasets["test"]),1)),
            "val"   : np.zeros((len(datasets["val"]),1)),
        }            
        colors = ["navy"]
        labels = ["energy"]
        ncomp = 1
        get_real = lambda x : x.energy
    else :
        raise ValueError("not implemented yet")

    #####################
    print("\tComputing the predictions for the datasets ...")
    pp_folder = "post-processing"

    pred = deepcopy(real)

    for name in ["train","val","test"]:
        for n,X in enumerate(datasets[name]):
            
            del X.edge_vec
            real[name][n,:] = get_real(X).detach().numpy()
            pred[name][n,:] = model(X).detach().numpy()

        if not os.path.exists(pp_folder):
            os.mkdir(pp_folder)

        np.savetxt("{:s}/pred.{:s}.txt".format(pp_folder,name),pred[name])
        np.savetxt("{:s}/real.{:s}.txt".format(pp_folder,name),real[name])

    #####################
    # plot correlation
    print("\tComputing the correlations ...")
    fig, axs = plt.subplots(ncols=3,figsize=(15,5))
    for n,name in enumerate(datasets.keys()):
        for xyz,c,l in zip(np.arange(ncomp),colors,labels):
            x = real[name][:,xyz]
            y = pred[name][:,xyz]
            axs[n].scatter(x,y,marker=".",color=c,label=l,s=0.5)

        axs[n].title.set_text(name) 
        lgnd = axs[n].legend()
        axs[n].grid()
        axs[n].set_aspect(1)
        plot_bisector(axs[n])

        # lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        for handle in lgnd.legend_handles:
            handle.set_sizes([10.0])

    plt.tight_layout()
    plt.savefig("{:s}/correlation.pdf".format(pp_folder))
    #plt.show()

    #####################
    # compute loss
    print("\tComputing the loss for the datasets ...")
    loss_func = model.loss()
    
    loss = pd.DataFrame(columns=["train","test","val"],index=["loss","RMSE","MSE"])
    for name in ["train","test","val"] :
        x = torch.from_numpy(pred[name])
        y = torch.from_numpy(real[name])
        loss.at["loss",name] = float(loss_func(x,y))
        loss.at["MSE",name]  = np.sqrt(loss.at["loss",name])
        #info.at["corr"] = pearsonr(pred[name],real[name]).correlation

    loss_file = "{:s}/stats.csv".format(pp_folder)
    loss.to_csv(loss_file,index=True)

    #####################
    # compute correlation    
    corr = pd.DataFrame(columns=["train","test","val"],index=["x","y","z"])
    for name in ["train","test","val"] :
        x = pred[name]
        y = real[name]
        for n,i in enumerate(["x","y","z"]) :
            corr.at[i,name] = pearsonr(x[:,n],y[:,n]).correlation
    
    corr_file = "{:s}/corr.csv".format(pp_folder)
    corr.to_csv(corr_file)

    #####################
    # save output
    outfile = "post-processing.json"
    print("\tWriting summary information to '{:s}'".format(outfile))
    output = {
        "parameters" : os.path.abspath(best_parameters),
        "folder"     : pp_folder,
        "stats"      : loss_file,
        "corr"       : corr_file,
    }

    with open(outfile, "w") as json_file:
        json.dump(output, json_file, indent=4)

    print("\n\tJob done :)\n")

#####################

if __name__ == "__main__":
    main()