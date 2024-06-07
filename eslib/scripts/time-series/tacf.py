#!/usr/bin/env python
from ase.io import write
from eslib.formatting import esfmt
import numpy as np
from eslib.mathematics import tacf, reshape_into_blocks
import matplotlib.pyplot as plt 
from eslib.plot import hzero
from scipy.optimize import curve_fit

#---------------------------------------#
# Description of the script's purpose
description = "Compute the time autocorrelation function (TACF) of an array."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"  , **argv, required=True , type=str, help="txt/npy input file")
    parser.add_argument("-b" , "--blocks" , **argv, required=False, type=int, help="number of blocks (default: %(default)s)", default=10)
    parser.add_argument("-m" , "--method" , **argv, required=False, type=str, help="method (default: %(default)s)", default='class', choices=['class','function'])
    parser.add_argument("-o" , "--output" , **argv, required=False, type=str, help="txt/npy output file (default: %(default)s)", default='tacf.npy')
    parser.add_argument("-p" , "--plot"   , **argv, required=False, type=str, help="output file for the plot (default: %(default)s)", default='tacf.pdf')

    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    if args.input.endswith("npy"):
        data = np.load(args.input)
    else:
        data = np.loadtxt(args.input)
    print("done")
    print("\tdata shape: :",data.shape)

    #------------------#
    print("\tn. of blocks: ", args.blocks)
    data = reshape_into_blocks(data,args.blocks).T
    print("\tdata shape: :",data.shape)

    #------------------#
    print("\tremoving mean ... ",end="")
    data -= data.mean(axis=0)
    print("done")

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    if args.method == "function":
        autocorr = tacf(data)
    else:
        from eslib.classes.tcf import TimeAutoCorrelation
        obj = TimeAutoCorrelation(data)
        autocorr = obj.tcf()
    print("done")
    print("\tautocorr shape: :",autocorr.shape)

    #------------------#
    print("\n\tSaving TACF to file '{:s}' ... ".format(args.output), end="")
    if args.output.endswith("npy"):
        np.save(args.output,autocorr)
    else:
        np.savetxt(args.output,autocorr)
    print("done")

    #------------------#
    print("\n\tFitting the autocorrelation function with an exponential ... ", end="")

    mean = autocorr.mean(axis=1)
    std = autocorr.std(axis=1)
    x = np.arange(len(mean))

    # Define the exponential function to fit
    def exponential(x,tau):
        return np.exp(-x/tau)

    # Fit the data to the exponential function, providing the standard deviation as sigma
    popt, pcov = curve_fit(f=exponential, 
                           xdata=x, 
                           ydata=mean,
                           sigma=std)

    yfit = exponential(x, *popt)
    print("done")
    print("\ttau: {:.0f}".format(popt[0]))


    #------------------#
    if args.plot is not None:
        print("\n\tProducing autocorrelation plot ... ", end="")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8),sharex=True)

        for i in range(autocorr.shape[1]):
            axes[0].plot(autocorr[:, i], label=f'{i+1}')

        axes[1].plot(x[1:],mean[1:],color="blue",label='$\\mu$')
        ylow,yhigh = mean - std, mean + std
        axes[1].fill_between(x[1:],ylow[1:],yhigh[1:], color='gray', alpha=0.25, label='$\\mu\\pm\\sigma$')
        ylow,yhigh = mean - 2*std, mean + 2*std
        axes[1].fill_between(x[1:],ylow[1:],yhigh[1:], color='gray', alpha=0.15, label='$\\mu\\pm2\\sigma$')

        axes[1].plot(x[1:],yfit[1:],color="red",label='$e^{-x/\\tau}$',linewidth=1)

        for ax in axes:
            ax.grid()
            ax.set_xlim(1,x.max())
            hzero(ax)
            # ax.set_xlim(1,x.max())
            ax.set_xscale("log")
        
        axes[1].legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")
        axes[1].set_xlabel('x')

        # Add text box with the value of a variable
        textbox_text = "$\\tau\\approx{:.0f}$".format(popt[0]) 
        # plt.text(0.1, 0.05, textbox_text,  ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        plt.annotate(textbox_text, xy=(0.5, 0.95), xycoords='axes fraction', fontsize=10,
                ha='center', va='top', bbox=dict(facecolor='white', alpha=1,edgecolor="black"))
        plt.tight_layout()
        print("done")

        print("\tSaving plot to file '{:s}'... ".format(args.plot), end="")
        plt.savefig(args.plot)
        print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()