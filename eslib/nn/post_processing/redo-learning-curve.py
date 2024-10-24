#!/usr/bin/env python
import argparse
# import json5 as json
import json

# from scipy.stats import pearsonr
# from copy import deepcopy
# import time
import matplotlib.pyplot as plt
import numpy as np
# import os
# import torch
import pandas as pd
# from elia.nn.functions.functions import get_model
# from elia.functions import plot_bisector
# from elia.nn.dataset import make_dataloader
# from elia.nn.plot.plot import plot_learning_curves
# from matplotlib.ticker import MaxNLocator
# import mpld3
from mpld3 import plugins  # , utils

# from chart_studio.plotly import plotly as py
# import chart_studio.tools as tls

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
        "--bs", action="store", type=int,
        help="batch size"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--lr", action="store", type=float,
        help="learning rate"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--output", action="store", type=str,
        help="output file"
    )

    # # Argument for "instructions"
    # parser.add_argument(
    #     "--max_time", action="store", type=int, default=-1,
    #     help="max_time"
    # )


    return parser# .parse_args()

def connect(fig,points):
    try : 
        labels0 = [ 'point {0}'.format(i + 1) for i in range(len(points))]
        tooltip = plugins.PointLabelTooltip(points, labels0)
        plugins.connect(fig, tooltip)
    except: 
        print("some error")

def main():

    args = get_args()

    with open(args.training, 'r') as file:
        parameters = json.load(file)     

    tmp = parameters["output_folder"], parameters["name"], args.bs, args.lr

    loss_file = "{:s}/dataframes/{:s}.bs={:d}.lr={:.1e}.csv".format(*tmp)
    arrays = pd.read_csv(loss_file)

    train_loss  = arrays["train"]   if "train"   in arrays else None
    val_loss    = arrays["val"]     if "val"     in arrays else None
    train_loss2 = arrays["train-2"] if "train-2" in arrays else None
    errors      = arrays["std"]     if "std"     in arrays else None
    ratio       = arrays["ratio"]   if "ratio"   in arrays else None
    ratio2      = arrays["ratio-2"] if "ratio-2" in arrays else None

    fig,ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(train_loss))+1

    
    points = ax.plot(x,val_loss,  color="red" ,label="val",  marker=".",linewidth=0.7,markersize=2,linestyle="-")
    connect(fig,points)

    ax.plot(x,train_loss,color="navy",label="$\\mu$-train",marker=".",linewidth=0.7,markersize=2,linestyle="-")
    if errors is not None :
        # ax.errorbar(x,train_loss,errors,color="navy",alpha=0.5,linewidth=0.5)
        points = ax.plot(x,errors,color="purple",label="$\\sigma$-train",marker=".",linewidth=0.7,markersize=2,linestyle="-")
        connect(fig,points)
    if train_loss2 is not None :
        points = ax.plot(x,train_loss2,color="green",label="train$^*$",marker=".",linewidth=0.7,markersize=2,linestyle="-")
        connect(fig,points)

    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, which="both",ls="-")
    xlim = ax.get_xlim()
    ax.set_xlim(1,xlim[1])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a twin axis on the right with a log scale
    if ratio is not None or ratio2 is not None :
        ax2 = ax.twinx()

        # if ratio is not None :
        #     ax2.plot(x, ratio, color="black", label="val/train",marker=".", linestyle="dotted",linewidth=0.7,markersize=2)

        if ratio2 is not None :
            points = ax2.plot(x, ratio2, color="black", label="train*/val",marker=".", linestyle="dotted",linewidth=0.7,markersize=2)
            connect(fig,points)
            xlim = ax2.get_xlim()
            points = ax2.hlines(y=1,xmin=xlim[0],xmax=xlim[1],linestyle="--",linewidth=0.7,alpha=0.5,color="black")
            connect(fig,points)

        ax2.set_yscale("log")
        ax2.set_ylabel("ratio")
        # ax2.legend(loc="upper right")

    plt.tight_layout()

    # html_str = mpld3.fig_to_html(fig)
    # Html_file= open("test.html","w")
    # Html_file.write(html_str)
    # Html_file.close()

    # # plt.show()
    plt.savefig(args.output)
    # #plt.close()



    print("\n\tJob done :)")

#####################

if __name__ == "__main__":
    main()