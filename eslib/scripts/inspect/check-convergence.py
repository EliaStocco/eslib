#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist
from eslib.plot import hzero
matplotlib.use('Agg')

#---------------------------------------#
# Description of the script's purpose
description = "Check the convergence of energy or forces w.r.t. to a parameter."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-x" , "--x_axis"      , **argv, type=str, required=False  , help="keyword of the x-axis (default: %(default)s)", default="N")
    parser.add_argument("-y" , "--y_axis"      , **argv, type=str, required=True , help="keyword of the info/array to check")
    parser.add_argument("-xlim", "--xlim"  , type=flist, **argv, required=False, help="xlim (default: %(default)s)", default=[None,None])
    parser.add_argument("-ylim", "--ylim"  , type=flist, **argv, required=False, help="ylim (default: %(default)s)", default=[None,None])
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output plot file (default: %(default)s)", default="convergence.png")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    # atomic structures
    print("\n\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures) 
    print("\tn. of atomic structures: ",N)
    Natoms = structures.num_atoms()
    
    #------------------#
    print("\n\tExtracting x-axis ... ", end="")
    x = structures.get(args.x_axis,"info").astype(int)
    print("done")
    print("\tx.shape: ",x.shape)
    
    #------------------#
    print("\n\tExtracting y-axis ... ", end="")
    what = structures.search(args.y_axis)
    y = structures.get(args.y_axis,what=what)
    print("done")
    print("\ty.shape: ",y.shape)
    
    #------------------#
    if what == "info":
        df = pd.DataFrame(columns=["x","y"])
        df["x"] = x
        df["y"] = y
        
        df = df.sort_values(by="x")
        df["y"] -= df["y"].iloc[-1] # values w.r.t. the last one
        
        df["y"] *= 1000 # ev -> meV
        df["y"] /= Natoms # global to per atoms
        
        
        #------------------#
        print("\n\tPlotting ... ", end="")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["x"].to_numpy(), df["y"].to_numpy(), marker='o', linestyle='-')
        ax.set_xlabel(args.x_axis)
        ax.set_ylabel(f"{args.y_axis} [meV/atoms]")
        # ax.set_title(f"Convergence of {args.y_axis} w.r.t. {args.x_axis}")
        ax.grid(True)
        ax.set_xlim(*args.xlim)
        ax.set_ylim(*args.ylim)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        hzero(ax)
        plt.tight_layout()
        plt.savefig(args.output,dpi=300,bbox_inches='tight')
        print("done")
        
    elif what == "arrays":

        print("\n\tProcessing array data for statistical plot ... ", end="")
        
        from eslib.metrics import RMSEforces
        
        rmse = RMSEforces(y,y[-1][None,:,:])
        rmse *= 1000 # eV/ang -> meV/ang
        # y = y.reshape((y.shape[0], -1))
        
        # ytest = RMSE(y,y[-1][:,None])

        # # Flatten last two dimensions: (N, A, D) → (N, A*D)
        # # y = y.reshape(y.shape[0], -1)
        # y -= y[-1, :, :]  # subtract last row
        # y = np.linalg.norm(y,axis=2)
        # # y -= y[-1, :]  # subtract last row

        # Compute mean, min, max per structure (i.e., row)
        df = pd.DataFrame({
            "x": x,
            "y": rmse
            # "mean": np.mean(rmse, axis=0),
            # "min": np.min(rmse, axis=1),
            # "max": np.max(rmse, axis=1)
        })

        print("done")

        #------------------#
        print("\n\tPlotting ... ", end="")
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(df["x"].to_numpy(), df["y"].to_numpy(), color="black", marker="o")
        # ax.plot(df["x"].to_numpy(), df["min"].to_numpy(), label="Min", color="red", linestyle="--")
        # ax.plot(df["x"].to_numpy(), df["max"].to_numpy(), label="Max", color="green", linestyle="--")

        ax.set_xlabel(args.x_axis)
        ax.set_ylabel(args.y_axis + " [meV/ang]")
        ax.set_title(f"Convergence of {args.y_axis} w.r.t. {args.x_axis}")
        ax.grid(True)
        ax.set_xlim(*args.xlim)
        ax.set_ylim(*args.ylim)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        hzero(ax)
        # ax.legend()
        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print("done")

    else:
        raise ValueError("coding error")
        

if __name__ == "__main__":
    main()

