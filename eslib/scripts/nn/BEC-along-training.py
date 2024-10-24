#!/usr/bin/env python
import argparse
# import json5 as json
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.io import read

from eslib.nn.user import get_model

# from chart_studio.plotly import plotly as py
# import chart_studio.tools as tls

#####################

description = "Recompute the train and validation losses in 'eval' mode"


def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-w",
        "--what",
        action="store",
        type=str,
        help="what to do ('r'=read, 'c'=compute, 'n'=norm, 'a'=all, 'p'=plot)",
        default="a",
    )
    parser.add_argument(
        "-t",
        "--training",
        action="store",
        type=str,
        help="training input file",
        default="input.json",
    )
    parser.add_argument(
        "-i",
        "--instructions",
        action="store",
        type=str,
        help="model input file",
        default="instructions.json",
    )
    parser.add_argument(
        "-bs", "--batch_size", action="store", type=int, help="batch size"
    )
    parser.add_argument(
        "-lr", "--learning_rate", action="store", type=float, help="learning rate"
    )
    parser.add_argument(
        "-q",
        "--positions",
        action="store",
        type=str,
        help="nuclear coordinates (a.u.)",
        default="positions.xsf",
    )
    parser.add_argument(
        "-z", "--BEC", action="store", type=str, help="BEC file", default="bec.txt"
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        help="output file prefix",
        default="output.pdf",
    )

    return parser# .parse_args()


def main():
    args = get_args()

    if args.what == "a":
        args.what = "rcnp"

    with open(args.training, "r") as file:
        parameters = json.load(file)

    atoms = read(args.positions)
    BEC = np.loadtxt(args.BEC)

    tmp_files = {
        "indeces": os.path.normpath("{:s}.indeces.csv".format(args.output)),
        "Z": os.path.normpath("{:s}.Z".format(args.output)),
        "norm": os.path.normpath("{:s}.norm.txt".format(args.output)),
    }

    # read
    if "r" in args.what:
        tmp = (
            parameters["output_folder"],
            parameters["name"],
            args.batch_size,
            args.learning_rate,
        )

        par_folder = "{:s}/parameters/{:s}.bs={:d}.lr={:.1e}".format(*tmp)
        files = os.listdir(par_folder)

        # cycle over parameters files
        indeces = pd.DataFrame(
            columns=["n", "epoch", "file"], index=np.arange(len(files))
        )

        for n, file in enumerate(files):
            # overwrite 'file'
            file = os.path.join(par_folder, file)

            epoch = int(file.split("epoch=")[1].split(".")[0])

            indeces.at[n, "n"] = n
            indeces.at[n, "epoch"] = int(epoch)
            indeces.at[n, "file"] = file

        indeces = indeces.sort_values(by="epoch")
        indeces.to_csv(tmp_files["indeces"], index=False)

    # compute
    if "c" in args.what:
        model = None
        positions = None
        cell = None
        indeces = pd.read_csv(tmp_files["indeces"])
        N = len(indeces)
        allZ = np.full((N, *BEC.shape), np.nan)
        k = 0
        for _, row in indeces.iterrows():
            n = row["n"]
            file = row["file"]
            epoch = row["epoch"]

            print("\tn:{:>5d}/{:<5d}".format(k + 1, N), end="")
            print("| epoch:{:<5d}".format(epoch), end="")
            print("| file: {:s}".format(file), end="\r")

            if model is None:
                model = get_model(args.instructions, file)
                if model.pbc:
                    cell = np.asarray(atoms.cell).T
                positions = np.asarray(atoms.positions)
            else:
                checkpoint = torch.load(file)
                model.load_state_dict(checkpoint)
                model.eval()

            allZ[k] = model.get_jac(pos=positions, cell=cell)[0].numpy()

            k += 1

        np.save(tmp_files["Z"], allZ)

    # norm
    if "n" in args.what:
        allZ = np.load(tmp_files["Z"] + ".npy")

        dZ = allZ - BEC
        N = len(dZ)
        norm = np.full(N, np.nan)

        for n in range(N):
            norm[n] = np.linalg.norm(dZ[n])

        np.savetxt(tmp_files["norm"], norm)

    # plot
    if "p" in args.what:
        norm = np.loadtxt(tmp_files["norm"]) / atoms.get_global_number_of_atoms()

        indeces = pd.read_csv(tmp_files["indeces"])
        epochs = np.asarray(indeces["epoch"])
        if epochs.min() == 0:
            epochs += 1

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(epochs, norm, color="navy")

        ax.grid(True, which="both", ls="-")
        plt.title("$|Z_{ref} - Z_{epoch}|$")
        plt.xlabel("epochs")
        plt.ylabel("RMSE/N$_{atoms}$ [number]")
        plt.yscale("log")
        plt.xscale("log")
        plt.tight_layout()

        plt.savefig(os.path.normpath(args.output+".pdf"))

    print("\n\tJob done :)")


#####################

if __name__ == "__main__":
    main()
