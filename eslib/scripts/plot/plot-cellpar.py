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
from eslib.input import itype, str2bool

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
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-s" , "--sort"       , **argv, required=False, type=str2bool, help="whether to sort a,b,c,alpha,beta,gamma (default: %(default)s)", default=True)
    parser.add_argument("-k", "--keyword"  , type=str, **argv, required=False , help="index keyword (default: %(default)s)", default="index")
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
        structures = AtomicStructures.from_file(file=f,format=args.input_format,index=args.index)
        print("done")
        cellpars[n] = structures.call(lambda s: s.get_cell().cellpar())
        
    print("\n\tAnalyzing data ... ",end="")
    cols = ["a","b","c","alpha","beta","gamma"]
    mean = [f"{a}-mean" for a in cols]
    std = [f"{a}-std" for a in cols]
    err = [f"{a}-err" for a in cols]
    df = pd.DataFrame(columns=[args.keyword]+mean+std+err,dtype=float)
    for n in range(N):
        df.loc[n,args.keyword] = indices[n]
        data = np.asarray(cellpars[n])
        for i,c in enumerate(cols):
            # df.loc[n,c] = data[:,i].mean()
            df.loc[n,mean[i]] = data[:,i].mean()
            df.loc[n,std[i]] = data[:,i].std()
            df.loc[n,err[i]] = data[:,i].std()/np.sqrt(data.shape[0])
        if args.sort:
            
            for cc in [["a", "b", "c"],["alpha", "beta", "gamma"]]:
                cmean = [f"{c}-mean" for c in cc]
                cstd = [f"{c}-std" for c in cc]
                cerr = [f"{c}-err" for c in cc]
                ii = np.argsort(df.loc[n,cmean])
                df.loc[n,cmean] = df.loc[n,cmean][ii]
                df.loc[n,cstd] = df.loc[n,cstd][ii]
                df.loc[n,cerr] = df.loc[n,cerr][ii]
                pass
    print("done")
    
    print(f"\n\tSaving data to '{args.output}' ... ",end="")
    df.to_csv(args.output,index=False)
    print("done")
    
    if args.plot is not None:

        print(f"\n\tPlotting data to '{args.plot}' ... ", end="")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

        x = df[args.keyword].to_numpy(dtype=float).astype(float)

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

        axes[1].set_xlabel(args.keyword)
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
