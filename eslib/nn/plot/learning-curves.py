import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

    return parser.parse_args()

def main():
    
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
    pp_folder = "post-processing"

    for bs in args.bs :
        for lr in args.lr :
            ifile = "{:s}/{:s}.bs={:d}.lr={:.1e}.csv".format(pp_folder,parameters["name"], bs, lr)
            loss = pd.read_csv(ifile)
            
            title = "{:s}: bs={:d}, lr={:.1e}".format(parameters["name"], bs, lr)
            ofile = "{:s}/{:s}.bs={:d}.lr={:.1e}.pdf".format(pp_folder,parameters["name"], bs, lr)
    

            train_loss = np.asarray(loss["train"])
            val_loss = np.asarray(loss["val"])
            file = ofile 
            title=title
            opts=None
            train_loss2=np.asarray(loss["train-2"])

            fig,ax = plt.subplots(figsize=(10,4))
            x = np.arange(len(train_loss))+1

            ax.plot(x,val_loss,  color="red" ,label="val",  marker="x",linewidth=0.7,markersize=2)
            # ax.plot(x,train_loss,color="navy",label="train",marker=".",linewidth=0.7,markersize=2)
            ii = ~np.isnan(train_loss2)
            ax.plot(x[ii],train_loss2[ii],color="green",label="train$^*$",marker=".",linewidth=0.7,markersize=2,linestyle="--")

            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.grid(True, which="both",ls="-")
            plt.xlim(1,x.max())
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title(title)

            plt.tight_layout()
            plt.savefig(file)

    print("\n\tJob done :)")

#####################

if __name__ == "__main__":
    main()