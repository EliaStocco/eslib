#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from eslib.classes.signal import Signal
from eslib.formatting import esfmt
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Compute the variance/flucutation of a time series at varying time to see when the value converges."

TEST = True

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # I/O
    parser.add_argument("-i" , "--input"    , **argv, required=True , type=str  , help="txt/npy input file")
    parser.add_argument("-o" , "--output"   , **argv, required=False, type=str  , help="txt/npy output file (default: %(default)s)", default='var.txt')
    # computation
    parser.add_argument("-N" , "--lenght"   , **argv, required=False, type=int  , help="lenght of each block (default: %(default)s)", default=100)
    parser.add_argument("-at", "--axis_time", **argv, required=False, type=int  , help="axis along compute autocorrelation (default: %(default)s)", default=0)
    parser.add_argument("-as", "--axis_sum" , **argv, required=False, type=int  , help="axis along compute the sum (default: %(default)s)", default=1)
    # plot
    parser.add_argument("-p" , "--plot"     , **argv, required=False, type=str  , help="output file for the plot (default: %(default)s)", default='var.pdf')
    parser.add_argument("-dt", "--time_step", **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-tc", "--time_corr", **argv, required=False, type=float, help="correlation time [fs] (default: %(default)s)", default=None)
    # parser.add_argument("-tu", "--time_unit", **argv, required=False, type=str  , help="time unit of the plot (default: %(default)s)", default="picosecond")
    # parser.add_argument("-tm", "--tmax"     , **argv, required=False, type=float, help="max time in plot [fs] (default: %(default)s)", default=None)
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
    # squared_mean:np.ndarray      = np.zeros((Ntot,*np.take(data,0, axis=args.axis_time).shape))
    # mean_squared          = np.zeros_like(squared_mean)
    # partial_fluctuations = np.zeros_like(squared_mean)
    # fluctuations         = np.zeros_like(squared_mean)
    # fill with nan
    # [ a.fill(np.nan) for a in [squared_mean,mean_squared,fluctuations] ]

    # print("\n\tRemoving mean ... ",end="")
    # data -= np.mean(data,axis=args.axis_time,keepdims=True)
    # print("done")

    data2 = np.square(data)

    print("\n\tComputing the fluctuation/variance along the axis {:d}:".format(args.axis_time))
    fluctuations     = np.full_like(np.zeros((Ntot,*np.take(data,0, axis=args.axis_time).shape)),np.nan)
    fluctuations_std = np.full_like(fluctuations,np.nan)
    
    # start computing
    for n,s in enumerate(sizes):
        print("\t{:d}/{:d} ... ".format(n+1,Ntot),end="\r")

        block = np.take(data, np.arange(0,s), axis=args.axis_time)
        fluctuations[n]     = np.var(block,axis=args.axis_time)
        assert np.allclose(np.mean(np.square(block),axis=args.axis_time)-np.square(np.mean(block,axis=args.axis_time)),fluctuations[n]), "Something is wrong"

        # block2 = np.take(data2, np.arange(0,s), axis=args.axis_time)
        # fluctuations_std[n] = np.var(block2,axis=args.axis_time)
        # continue

        block2 = np.square(block)
        block4 = np.square(block2)
        N = block4.shape[args.axis_time]
        if args.time_corr is not None:
            Neff = N*args.time_step/args.time_corr
        else:
            Neff = N
        alpha = 3-Neff/(Neff-1)
        fluctuations_std[n] = np.mean(block4, axis=args.axis_time)+alpha*np.square(np.mean(block2, axis=args.axis_time))
        fluctuations_std[n] /= Neff

        # block = np.take(data, np.arange(s-args.lenght,s), axis=args.axis_time)
        # assert block.shape[args.axis_time] == args.lenght, "The shape of the block must be ({:d},{:d})".format(args.lenght,block.shape[args.axis_time])
        # squared_mean[n] = np.square(np.mean(block, axis=args.axis_time))
        # mean_squared[n] = np.mean(np.square(block), axis=args.axis_time)
        # partial_fluctuations[n] = mean_squared[n] - squared_mean[n]

        # tmp_SM = np.take(squared_mean, np.arange(0,n+1), axis=args.axis_time)
        # tmp_MQ = np.take(mean_squared, np.arange(0,n+1), axis=args.axis_time)
        # fluctuations[n] = np.mean(tmp_MQ-tmp_SM,axis=args.axis_time)

        # if TEST:
        #     block = np.take(data, np.arange(0,s), axis=args.axis_time)
        #     tmp = np.var(block,axis=args.axis_time)
        #     assert np.allclose(tmp,fluctuations[n]), "tmp and fluctuations must be the same"


    #------------------#
    fluctuations = np.asarray(fluctuations)
    print("\tfluctuations shape: ",fluctuations.shape)

    #------------------#
    print("\n\tComputing the sum along the axis {:d} ... ".format(args.axis_sum),end="")
    fluctuations = np.sum(fluctuations,axis=args.axis_sum)
    print("done")
    print("\tfluctuations shape: ",fluctuations.shape)

    if fluctuations.ndim == 1:
        print("\tReshaping data  ... ", end="")
        fluctuations = np.atleast_2d(fluctuations).T
        print("done")
        print("\tfluctuations shape: ",fluctuations.shape)        
    # assert fluctuations.ndim == 1, "fluctuations must be a 1D array"

    #------------------#
    print("\n\tSaving fluctuations to file {:s} ... ".format(args.output),end="")
    index = np.arange(Ntot)
    time = sizes*args.time_step/1000 # fs to ps
    tmp =  np.vstack((index,sizes,time,fluctuations.T)).T
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
    for n in range(fluctuations.shape[1]):
        ax.plot(time, fluctuations[:,n], color=colormap(n/(fluctuations.shape[1])), label=f"{n+1}")
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