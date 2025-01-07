#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.formatting import esfmt
from eslib.mathematics import mean_std_err2pandas
from eslib.plot import legend_options

#---------------------------------------#
description = "Computes the mean, standard deviation and standard error of some data + produce a plot."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"    , **argv, required=True , type=str  , help="input files")
    parser.add_argument("-a" , "--axis"     , **argv, required=False, type=int  , help='axis (default: %(default)s)', default=1)
    parser.add_argument("-o" , "--output"   , **argv, required=True , type=str  , help="output file")
    parser.add_argument("-p" , "--plot"     , **argv, required=None , type=str  , help="plot file (default: %(default)s)", default=None)
    parser.add_argument("-dt", "--time_step", **argv, required=False, type=float, help="time step (default: %(default)s)", default=1)
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    data = np.loadtxt(args.input)
    print("done")
    print("\tshape: ",data.shape)
    assert data.ndim == 2, "The input file must be a 2D array"
    
    #------------------#
    print("\n\tComputing the mean, standard deviation and standard error ... ", end="")
    df = mean_std_err2pandas(data, axis=args.axis)
    print("done")
    
    #------------------#
    print("\n\tWriting the dataframe to file '{:s}' ... ".format(args.output), end="")
    df.to_csv(args.output)
    print("done")
    
    #------------------#
    if args.plot is not None:
        print("\n\tSaving plot to file '{:s}' ... ".format(args.plot), end="")
        fig,ax = plt.subplots(figsize=(4,3))
        time = np.arange(len(df))*args.time_step
        ax.plot(time,df["mean"],label="mean")
        ax.fill_between(time,df["mean"]-df["err"],df["mean"]+df["err"],alpha=0.6)
        ax.fill_between(time,df["mean"]-df["std"],df["mean"]+df["std"],alpha=0.3)
        ax.legend(**legend_options)
        plt.tight_layout()                
        plt.savefig(args.plot,bbox_inches="tight",dpi=300)  
        print("done")
    
    

#---------------------------------------#
if __name__ == "__main__":
    main()