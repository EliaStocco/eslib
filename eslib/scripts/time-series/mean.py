#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from eslib.classes.signal import Signal
from eslib.formatting import esfmt
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Compute the mean of a time series at varying time to see when the value converges."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # I/O
    parser.add_argument("-i" , "--input"    , **argv, required=True , type=str  , help="txt/npy input file")
    parser.add_argument("-o" , "--output"   , **argv, required=False, type=str  , help="txt/npy output file (default: %(default)s)", default='mean.txt')
    # computation
    parser.add_argument("-N" , "--lenght"   , **argv, required=False, type=int  , help="lenght of each block (default: %(default)s)", default=100)
    parser.add_argument("-at", "--axis_time", **argv, required=False, type=int  , help="axis along compute autocorrelation (default: %(default)s)", default=0)
    # plot
    parser.add_argument("-p" , "--plot"     , **argv, required=False, type=str  , help="output file for the plot (default: %(default)s)", default='mean.pdf')
    parser.add_argument("-dt", "--time_step", **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    data:np.ndarray = Signal.from_file(file=str(args.input))
    print("done")
    print("\tdata shape: ",data.shape) 

    

    #------------------#   
    Ntot = int(data.shape[args.axis_time]/args.lenght)
    sizes = np.arange(1,Ntot+1)*args.lenght
    assert len(sizes) == Ntot, "The total number of sizes must be equal to the number of blocks"
    assert sizes[-1] <= data.shape[args.axis_time], "The last size must be smaller than the total size"
    print("\n\tData will be partitioned into {:d} arrays of increasing size of {:d}.".format(Ntot,args.lenght))

    #------------------#
    print("\n\tComputing the fluctuation/variance along the axis {:d}:".format(args.axis_time))
    mean = np.full_like(np.zeros((Ntot,*np.take(data,0, axis=args.axis_time).shape)),np.nan)
    std  = np.full_like(mean,np.nan)
    
    # start computing
    for n,s in enumerate(sizes):
        print("\t{:d}/{:d} ... ".format(n+1,Ntot),end="\r")

        block = np.take(data, np.arange(0,s), axis=args.axis_time)
        mean[n]     = np.mean(block,axis=args.axis_time)
        std[n]      = np.std(block,axis=args.axis_time)

    #------------------#
    mean = np.asarray(mean)
    print("\tmean shape: ",mean.shape)

    #------------------#
    if mean.ndim == 1:
        print("\tReshaping data  ... ", end="")
        mean = np.atleast_2d(mean).T
        print("done")
        print("\tmean shape: ",mean.shape)        
    # assert mean.ndim == 1, "mean must be a 1D array"

    #------------------#
    print("\n\tSaving mean to file {:s} ... ".format(args.output),end="")
    index = np.arange(Ntot)
    time = sizes*args.time_step/1000 # fs to ps
    tmp =  np.vstack((index,sizes,time,mean.T)).T
    tmp = Signal(tmp)
    if str(args.output).endswith("txt"):
        header = \
            f"Col 1: index\n" +\
            f"Col 2: lenght of the block\n" +\
            f"Col 3: maximum time considered in the block [ps]\n" +\
            f"Col 4 onwards: fluctuation/variance of the block"
            # f"Col 5: standard deviation of the fluctuation of the block"
        tmp.to_file(file=args.output,header=header)
    else:
        tmp.to_file(file=args.output)
    print("done")

    #------------------#
    print("\n\tSaving the plot to file {:s} ... ".format(args.plot),end="")
    fig, ax = plt.subplots(1, figsize=(6, 4))
    colormap = plt.cm.viridis
    for n in range(mean.shape[1]):
        ax.plot(time, mean[:,n], color=colormap(n/(mean.shape[1])), label=f"{n+1}")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("fluctuation [arb. units]")
    ax.grid()
    ax.set_xlim(0, time[-1])
    #ax.set_ylim(0, None)
    ax.legend(framealpha=1, edgecolor="black", facecolor='white')
    plt.tight_layout()
    plt.savefig(args.plot)
    print("done")

    #------------------#
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()