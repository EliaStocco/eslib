#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('Agg')  # must come first


from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functions import extract_number_from_filename
from eslib.io_tools import pattern2sorted_files

#---------------------------------------#
# Description of the script's purpose
description = "Plot a time series of a 'info' of a 'extxyz' file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , type=str, **argv, required=True , help='input extxyz file')
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n", "--name"  , type=str, **argv, required=False , help="index name (default: %(default)s)", default="index")
    parser.add_argument("-o", "--output", type=str, **argv, required=False, help="CSV output file (default: %(default)s)", default="cellpar.csv")
    parser.add_argument("-p", "--plot", type=str, **argv, required=False, help="plot output file (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    files = pattern2sorted_files(args.input)
    N = len(files)
    print(f"\tFound {N} files matching the pattern '{args.input}'")
    indices = [extract_number_from_filename(f) for f in files]
    cellpars = [None]*N
    for n,f in enumerate(files):
        print(f"\tReading file {f} ... ",end="")
        structures = AtomicStructures.from_file(file=f,format=args.input_format)
        print("done")
        cellpars[n] = structures.call(lambda s: s.get_cell().cellpar())
        
    print("\n\tAnalyzing data ... ",end="")
    cols = ["a","b","c","alpha","beta","gamma"]
    mean = [f"{a}-mean" for a in cols]
    std = [f"{a}-std" for a in cols]
    err = [f"{a}-err" for a in cols]
    df = pd.DataFrame(columns=[args.name]+mean+std+err,dtype=float)
    for n in range(N):
        df.loc[n,args.name] = indices[n]
        data = np.asarray(cellpars[n])
        for i,c in enumerate(cols):
            # df.loc[n,c] = data[:,i].mean()
            df.loc[n,mean[i]] = data[:,i].mean()
            df.loc[n,std[i]] = data[:,i].std()
            df.loc[n,err[i]] = data[:,i].std()/np.sqrt(data.shape[0])
    print("done")
    
    print(f"\n\tSaving data to '{args.output}' ... ",end="")
    df.to_csv(args.output,index=False)
    print("done")
    
    if args.plot is not None:

        print(f"\n\tPlotting data to '{args.plot}' ... ", end="")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

        x = df[args.name].to_numpy(dtype=float).astype(float)

        # --------- First plot: a, b, c ----------
        for c in ["a", "b", "c"]:
            y = df[f"{c}-mean"].to_numpy(dtype=float).astype(float)
            err = df[f"{c}-err"].to_numpy(dtype=float).astype(float)

            axes[0].plot(x, y, label=c)
            axes[0].fill_between(x, y - err, y + err, alpha=0.3)

        axes[0].set_ylabel("Lattice parameters (Å)")
        axes[0].set_title("a, b, c")
        axes[0].legend()
        axes[0].grid(True)

        # --------- Second plot: alpha, beta, gamma ----------
        for c in ["alpha", "beta", "gamma"]:
            y = df[f"{c}-mean"].to_numpy(dtype=float).astype(float)
            err = df[f"{c}-err"].to_numpy(dtype=float).astype(float)

            axes[1].plot(x, y, label=c)
            axes[1].fill_between(x, y - err, y + err, alpha=0.3)

        axes[1].set_xlabel(args.name)
        axes[1].set_ylabel("Angles (deg)")
        axes[1].set_title("alpha, beta, gamma")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(args.plot)
        plt.close(fig)

        print("done")
    

    return

if __name__ == "__main__":
    main()
