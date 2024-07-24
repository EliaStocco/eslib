#!/usr/bin/env python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tomlkit import comment 
from eslib.mathematics import tacf, reshape_into_blocks
from eslib.plot import hzero
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool
from eslib.formatting import esfmt
from eslib.classes.tcf import TimeAutoCorrelation, compute_spectrum, get_freq
# from eslib.classes.spectrum import Spectrum

#---------------------------------------#
# Description of the script's purpose
description = "Compute the time autocorrelation function (TACF) of a dipole time series and compute the Infra Red (IR) spectrum."
documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of dipole."

alpha = 0.5

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # I/O
    parser.add_argument("-i" , "--input"      , **argv, required=True , type=str     , help="txt/npy input file")
    parser.add_argument("-o" , "--output"     , **argv, required=False, type=str     , help="txt/npy output file (default: %(default)s)", default='tacf.npy')
    # Calculations
    parser.add_argument("-dt", "--time_step"  , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-d" , "--derivative" , **argv, required=False, type=str2bool, help="compute derivative of the input data (default: %(default)s)", default=True)
    parser.add_argument("-b" , "--blocks"     , **argv, required=False, type=int     , help="number of blocks (default: %(default)s)", default=10)
    parser.add_argument("-ac", "--axis_corr"  , **argv, required=False, type=int     , help="axis along compute autocorrelation (default: %(default)s)", default=1)
    parser.add_argument("-am", "--axis_mean"  , **argv, required=False, type=int     , help="axis along compute mean (default: %(default)s)", default=2)
    parser.add_argument("-rm", "--remove_mean", **argv, required=False, type=str2bool, help="whether to remove the e mean (default: %(default)s)", default=False)
    parser.add_argument("-m" , "--method"     , **argv, required=False, type=str     , help="method (default: %(default)s)", default='class', choices=['class','function'])
    # Plot
    parser.add_argument("-p" , "--plot"       , **argv, required=False, type=str     , help="output file for the plot (default: %(default)s)", default='tacf.pdf')
    parser.add_argument("-tm", "--tmax"       , **argv, required=False, type=float   , help="max time in TACF plot [fs] (default: %(default)s)", default=500)
    parser.add_argument("-f" , "--fit"        , **argv, required=False, type=str2bool, help="whether to fit the TACF with an exponential (default: %(default)s)", default=True)
    # Window and padding
    parser.add_argument("-w"   , "--window"   , **argv, required=False, type=str     , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt"  , "--window_t" , **argv, required=False, type=int     , help="time span of the window [fs] (default: %(default)s)", default=10)
    # Infrared Spectrum
    parser.add_argument("-ir" , "--infrared"  , **argv, required=False , type=str    , help="output file with the Infrared spectrum (default: %(default)s)", default=None)
    parser.add_argument("-pad" , "--padding"  , **argv, required=False, type=int     , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    # Plot
    parser.add_argument("-pir" , "--plot_infrared", **argv, required=False, type=str , help="output file for the plot (default: %(default)s)", default='IR.pdf')
    parser.add_argument("-mf", "--max_freq"       , **argv, required=False, type=float, help="max frequency in IR plot [THz] (default: %(default)s)", default=500)
    parser.add_argument("-fu", "--freq_unit"      , **argv, required=False, type=str , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="THz")
    parser.add_argument("-ms", "--marker_size"    , **argv, required=False, type=float , help="marker size (default: %(default)s)", default=0)
    return parser

#---------------------------------------#
# Define the exponential function to fit
def exponential(x,tau):
    """
    A function that calculates the exponential of x divided by tau.

    Parameters:
        x (float): The input value.
        tau (float): The time constant.

    Returns:
        float: The exponential of -x/tau.
    """
    return np.exp(-x/tau)
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    data:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    print("\tdata shape: ",data.shape)

    #------------------#
    if args.blocks > 0 :
        print("\n\tN. of blocks: ", args.blocks)
        print("\tBuilding blocks ... ",end="")
        data = reshape_into_blocks(data,args.blocks)# .T
        print("done")
        print("\tdata shape: ",data.shape)

    #------------------#
    if args.derivative:
        print("\n\tComputing the derivative ... ",end="")
        data = np.gradient(data,axis=args.axis_corr)/args.time_step
        print("done")
        print("\tdata shape: ",data.shape)

    #------------------#
    if args.remove_mean:
        print("\n\tRemoving mean ... ",end="")
        data -= np.mean(data,axis=args.axis_corr,keepdims=True)
        print("done")

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    if args.method == "function":
        autocorr = tacf(data)
    else:
        obj = TimeAutoCorrelation(data)
        autocorr = obj.tcf(axis=args.axis_corr)
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if args.axis_mean is not None:
        print("\n\tComputing the mean along the axis {:d} ... ".format(args.axis_mean),end="")
        autocorr = np.mean(autocorr,axis=args.axis_mean)
        print("done")
        print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if autocorr.ndim == 1:
        print("\tReshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)# .T
        
        print("done")
        print("\tdata shape: ",data.shape)

    #------------------#
    print("\n\tComputing the average over the trajectories:")
    mean = autocorr.mean(axis=0)
    std = autocorr.std(axis=0)
    x = np.arange(autocorr.shape[1])*args.time_step
    print("\t   x shape: ",x.shape)
    print("\tmean shape: ",mean.shape)
    print("\t std shape: ",std.shape)

    #------------------#
    if args.fit:
        print("\n\tFitting the autocorrelation function with an exponential ... ",end="")

        # Fit the data to the exponential function, providing the standard deviation as sigma
        popt, pcov = curve_fit(f=exponential, 
                            xdata=x, 
                            ydata=mean,
                            sigma=std)

        yfit = exponential(x, *popt)
        print("done")
        print("\ttau: {:.0f}fs".format(popt[0]))

    #------------------#
    if args.plot is not None:
        width = 8 if args.window == "none" else 12
        nplots = 2 if args.window == "none" else 3
        fig, axes = plt.subplots(nplots, 1, figsize=(10, width),sharex=True)

        for i in range(autocorr.shape[0]):
            y = autocorr[i]
            axes[0].plot(x,y, label=f'{i+1}')

        axes[1].plot(x,mean,color="blue",label='$f$')
        ylow,yhigh = mean - std, mean + std
        axes[1].fill_between(x,ylow,yhigh, color='gray', alpha=alpha, label='$f\\pm\\sigma$')
        ylow,yhigh = mean - 2*std, mean + 2*std
        axes[1].fill_between(x,ylow,yhigh, color='gray', alpha=alpha/2., label='$f\\pm2\\sigma$')

        if args.fit:
            axes[1].plot(x,yfit,color="red",label='$e^{-x/\\tau}$',linewidth=1)

        if args.fit:
            # Add text box with the value of a variable
            textbox_text = "$\\tau\\approx{:.0f}$fs".format(popt[0]) 
            # plt.text(0.1, 0.05, textbox_text,  ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            axes[1].annotate(textbox_text, xy=(0.5, 0.95), xycoords='axes fraction', fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=1,edgecolor="black"))
        axes[1].legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")

                

    #------------------#
    if args.window != "none" :
        print("\n\tApplying the '{:s}' window ... ".format(args.window),end="")
        #Define window to be used with DCT below.
        func = getattr(np, args.window)
        window = np.zeros(autocorr.shape[args.axis_corr])
        M = int(args.window_t / args.time_step)
        window[:M] = func(2*M)[M:]
        # window = np.atleast_2d(window)
        print("done")
        print("\twindow shape: ",window.shape)

        
        # raw_autocorr = np.atleast_2d(raw_autocorr)

        if args.plot is not None:
            # raw TACF
            axes[2].plot(x,mean,color="blue",label='$f$')
            # ylow,yhigh = mean - std, mean + std
            # axes[2].fill_between(x,ylow,yhigh, color='gray', alpha=alpha)
            # cleaned TACF
            autocorr = autocorr * window
            mean = autocorr.mean(axis=0)
            std = autocorr.std(axis=0)
            axes[2].plot(x,mean,color="red",label='$f_{\\rm cleaned}$')
            ylow,yhigh = mean - std, mean + std
            axes[2].fill_between(x,ylow,yhigh, color='gray', alpha=alpha, label='$f_{\\rm cleaned}\\pm\\sigma$')
            axes[2].legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")
            # window
            axes[2].plot(x,window,  color="black",label='window')


    #------------------#
    if args.plot is not None:
        print("\n\tSaving plot to file '{:s}'... ".format(args.plot), end="")
        for ax in axes:
            ax.grid()
            ax.set_xlim(0,min(args.tmax,x.max()))
            hzero(ax)
            # ax.set_xlim(1,x.max())
            # ax.set_xscale("log")
        axes[-1].set_xlabel('time [fs]')
        plt.suptitle("Time Autocorrelation Function")
        plt.tight_layout()
        plt.savefig(args.plot)
        print("done")
    

    #------------------#
    print("\n\tConverting autocorr into PhysicalTensor ... ", end="")
    autocorr = PhysicalTensor(autocorr)
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    print("\tSaving TACF to file '{:s}' ... ".format(args.output), end="")
    autocorr.to_file(file=args.output)
    print("done")

    #------------------#
    if args.infrared is not None:
        print("\n\tComputing the spectra ... ", end="")
        spectrum = compute_spectrum(autocorr,axis=args.axis_corr,pad=args.padding)
        print("done")
        print("\tspectrum shape: :",spectrum.shape)

        #------------------#
        print("\n\tNormalizing the spectra ... ", end="")
        factor   = np.max(spectrum,axis=args.axis_corr)[:, np.newaxis]
        spectrum = np.divide(spectrum,factor)
        std      = np.divide(std,factor)
        print("done")
        print("\tspectrum shape: :",spectrum.shape)    

        assert np.allclose(np.max(spectrum,axis=args.axis_corr),1), "the spectra are not normalized"

        #------------------#
        print("\n\tComputing the average over the trajectories ... ", end="")
        std:np.ndarray = spectrum.std(axis=0)
        spectrum:np.ndarray = spectrum.mean(axis=0)
        print("done")
        print("\tspectrum shape: :",spectrum.shape)

        assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"


        #------------------#
        print("\n\tComputing the frequencies ... ", end="")
        freq = get_freq(dt=args.time_step, N=len(spectrum),output_units=args.freq_unit)
        print("done")

        print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.infrared), end="")
        tmp =  np.vstack((freq,spectrum,std)).T
        assert tmp.ndim == 2, "thsi array should have 2 dimensions"
        assert tmp.shape[1] == 3, "this array should have 3 columns"
        tmp = PhysicalTensor(tmp)
        if str(args.infrared).endswith("txt"):
            header = \
                f"Col 1: frequency in {args.freq_unit}\n" +\
                f"Col 2: normalized spectrum\n" +\
                f"Col 3: std (over trajectories) of the spectrum "
            tmp.to_file(file=args.infrared,header=header)
        else:
            tmp.to_file(file=args.infrared)
        
        del tmp
        print("done")


        #------------------#
        if args.plot_infrared:
            print("\tPreparing plot ... ", end="")
            fig, ax = plt.subplots(1, figsize=(6, 4))
            # y = np.linalg.norm(spectrum.mean(axis=args.axis_corr),axis=-1)
            # y /= np.max(y)
            # ax.plot(freq,y,label="raw",color="red")
            ax.plot(freq,spectrum, label="$\\rm S\\left(\\omega\\right)$",color="blue", marker='.', markerfacecolor='blue', markersize=args.marker_size)
            ylow,yhigh = spectrum - std, spectrum + std
            ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.8, label='$\\rm S\\left(\\omega\\right)\\pm\\sigma\\left(\\omega\\right)$')
            # ylow,yhigh = spectrum - 2*std, spectrum + 2*std
            # ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.5, label='$\\pm2\\sigma$')
            ax.legend(loc="upper left",facecolor='white', framealpha=1,edgecolor="black")
            ax.set_xlim(0,args.max_freq)
            ax.set_ylim(0,None)
            ax.set_yticks(np.arange(0,1.001,0.2))
            ax.set_xlabel("frequency [THz]")
            ax.set_ylabel("spectrum [arb. units]")
            ax.grid()
            plt.tight_layout()
            print("done")

            #------------------#
            print("\tSaving plot to file '{:s}'... ".format(args.plot_infrared), end="")
            plt.savefig(args.plot_infrared)
            # plt.show()
            print("done")
        

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()