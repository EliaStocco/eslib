#!/usr/bin/env python
import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from eslib.formatting import esfmt
from eslib.mathematics import reshape_into_blocks, histogram_along_axis
from eslib.input import ilist

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Shannon entropy based on the SOAP descriptors."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-x"  , "--soap_descriptors", type=str     , required=True , **argv, help="file with the SOAP descriptors")
    parser.add_argument("-b"  , "--n_bins"          , type=ilist     , required=False, **argv, help="number of bins (default: [1000])", default=[1000])
    parser.add_argument("-n"  , "--block_length"    , type=int     , required=False, **argv, help="length of each block (default: 100)", default=100)
    parser.add_argument("-o"  , "--output"          , type=str     , required=False, **argv, help="output file with the Kullback-Leibler divergence (default: entropy.csv)", default='entropy.csv')
    parser.add_argument("-p"  , "--plot"            , type=str     , required=False, **argv, help="plot (default: 'entropy.pdf')", default='entropy.pdf')
    return parser



# @njit(parallel=True)
# def histogram_along_axis(data: np.ndarray, bins: int, axis: int) -> np.ndarray:
#     """
#     Compute the histogram of a numpy array along a specific axis.

#     Parameters:
#         data (numpy.ndarray): The input array.
#         bins (int): The number of bins for the histogram.
#         axis (int): The axis along which to compute the histogram.

#     Returns:
#         numpy.ndarray: The histogram values along the specified axis.

#     Example:
#         >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#         >>> bins = 5
#         >>> axis = 1
#         >>> hist = histogram_along_axis(data, bins, axis)
#         >>> print(hist)
#         array([[1, 1, 1, 0, 0],
#                [1, 1, 1, 0, 0],
#                [1, 1, 1, 0, 0]])
#     """
#     # Transpose the array to move the specified axis to the first position
#     data_transposed = np.transpose(data, np.roll(np.arange(data.ndim), -axis))

#     # Compute the histogram along the first axis
#     range = (np.min(data), np.max(data))
#     hist = np.zeros((bins,) + data_transposed.shape[1:], dtype=np.int64)
#     for i in prange(data_transposed.shape[1]):
#         hist[:, i] = np.histogram(data_transposed[:, i], bins=bins, range=range)[0]

#     # Transpose the result back to the original shape
#     hist_transposed = np.transpose(hist, np.roll(np.arange(hist.ndim), axis))

#     return hist_transposed

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    # assert args.n_bins < args.block_length

    #------------------#
    print("\n\tReading SOAP descriptors from file '{:s}' ... ".format(args.soap_descriptors),end="")
    if str(args.soap_descriptors).endswith("npy"):
        X = np.load(args.soap_descriptors)
    elif str(args.soap_descriptors).endswith("txt"):
        X = np.loadtxt(args.soap_descriptors)
    print("done")
    print("\tSOAP.shape: ",X.shape)

    #------------------#
    N = X.shape[0] // args.block_length
    print("\n\tDividing the SOAP descriptors into '{:d}' blocks ... ".format(N),end="")
    Xblocks = reshape_into_blocks(X,N=N)
    print("done")
    print("\tSOAP.shape: ",Xblocks.shape)
    print("\tn. of discarded structures: ",X.shape[0]-Xblocks.shape[0]*Xblocks.shape[1])
    X = xr.DataArray(Xblocks.copy(),dims=["dataset","structures","descriptors"])
    del Xblocks

    #------------------#
    print("\n\tComputing the histogram of the SOAP descriptors ... ",end="")
    hist = [None]*len(args.n_bins)
    for n,nbins in enumerate(args.n_bins):
        # nbins = args.n_bins[n]
        hist[n] = histogram_along_axis(X.data,nbins,X.dims.index("structures"))
    # hist = np.asarray(hist)
    # hist = np.moveaxis(hist, 0, -1)
    hist = [ xr.DataArray(h,dims=("dataset","x","descriptors")) for h in hist]
    print("done")
    print("\thist.shape: ",hist[0].shape)

    #------------------#
    print("\n\tAccumulatig histograms ... ",end="")
    hist = [ np.cumsum(h, axis=h.dims.index("dataset")) for h in hist ] 
    print("done")
    print("\thist.shape: ",hist[0].shape)

    for h in hist:
        h.data[h.data > 0] = 1
    hist = [ xr.DataArray(h) for h in hist] 

    #------------------#
    print("\n\tComputing the Shannon entropy ... ",end="")
    shape = hist[0].shape
    shape = shape[:1] + shape[2:] + (len(args.n_bins),)
    SE = np.full(shape,np.nan)

    for b,h in enumerate(hist): # cycle over nbins
        # for n,p in h.coords["dataset"].items(): # cycle over dataset
        for n,p in enumerate(h): # cycle over dataset
            # p:xr.DataArray = hist.isel(dataset=n)
            SE[n,:,b] = entropy(p.data,axis=p.dims.index("x"))

    SE = xr.DataArray(SE,dims=("dataset","descriptors","bins"))
    SE = SE.mean('descriptors')
    print("done")
    print("\tShannon entropy shape: ",SE.shape)

    #------------------#
    df = SE.to_dataframe(name="entropy").reset_index()
    df['size'] = (df['dataset']+1)*args.block_length
    df['bins'] = args.n_bins[df['bins']]
    print("\n\tSaving results to file '{:s}' ... ".format(args.output),end="")
    df.to_csv(args.output,index=False)
    print("done")

    #------------------#
    # Group DataFrame by 'nbins' column
    groups = df.groupby('bins')

    fig,axes = plt.subplots(ncols=1,nrows=2,figsize=(10,6),sharex=True)
    for name, group in groups:
        label = list(group['bins'])[0]
        axes[0].plot(group['size'], group['entropy'], marker='.', label=label)
        ref = float(group[group['size'] == max(group['size'])]['entropy'])
        axes[1].plot(group['size'][:-2], -(group['entropy']-ref)[:-2], marker='.', label=name)
    
    for ax,loc in zip(axes,['upper left','upper right']):
        ax.legend(title="n. bins",facecolor='white', framealpha=1,edgecolor="black",loc=loc)
        ax.grid(True)

    axes[1].set_xlabel('Dataset size')
    # axes[1].set_xscale('log')
    # axes[1].set_yscale('log')
    axes[0].set_ylabel('H')
    axes[1].set_ylabel('-$\\Delta$H')
    axes[0].set_title('Shannon entropy H')
    plt.tight_layout()

    print("\n\tSaving plot to file '{:s}' ... ".format(args.plot),end="")
    plt.savefig(args.plot)
    print("done")

    return 0 
   
#---------------------------------------#
if __name__ == "__main__":
    main()

# { 
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/descriptors/shannon.py",
#             "cwd" : "/home/stoccoel/google-personal/simulations/LiNbO3/ML/LiNbO3-oxn/original-data",
#             "args" : ["-x","soap.npy","-b","[10,100,300,600,1000]"],
#             "console": "integratedTerminal",
#             "justMyCode": true,
#         }
#     ]
# }