#!/usr/bin/env python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from eslib.mathematics import tacf
from eslib.plot import hzero
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool, flist
from eslib.formatting import esfmt
from eslib.classes.tcf import TimeAutoCorrelation, compute_spectrum, get_freq
from eslib.tools import convert
# from eslib.classes.spectrum import Spectrum

#---------------------------------------#
# Description of the script's purpose
description = "Compute the frequency dependent electric susceptibility (defined as dielectric constant -1) from a dipole time series."
# documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of dipole."

alpha = 0.5

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # I/O
    parser.add_argument("-i" , "--input"      , **argv, required=True , type=str     , help="txt/npy input file")
    parser.add_argument("-o" , "--output"     , **argv, required=False, type=str     , help="txt/npy output file (default: %(default)s)", default='dielectric.txt')
    # Calculations
    parser.add_argument("-dt", "--time_step"  , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-d" , "--derivative" , **argv, required=False, type=str2bool, help="use the derivative of the input data to perform the Fourier transform (default: %(default)s)", default=True)
    # parser.add_argument("-b" , "--blocks"     , **argv, required=False, type=int     , help="number of blocks (default: %(default)s)", default=10)
    parser.add_argument("-ac", "--axis_corr"  , **argv, required=False, type=int     , help="axis along compute autocorrelation (default: %(default)s)", default=1)
    parser.add_argument("-am", "--axis_mean"  , **argv, required=False, type=int     , help="axis along compute mean (default: %(default)s)", default=2)
    parser.add_argument("-rm", "--remove_mean", **argv, required=False, type=str2bool, help="whether to remove the mean (default: %(default)s)", default=True)
    # parser.add_argument("-m" , "--method"     , **argv, required=False, type=str     , help="method (default: %(default)s)", default='class', choices=['class','function'])
    # Plot
    # parser.add_argument("-p" , "--plot"       , **argv, required=False, type=str     , help="output file for the plot (default: %(default)s)", default='tacf.pdf')
    # parser.add_argument("-tm", "--tmax"       , **argv, required=False, type=float   , help="max time in TACF plot [fs] (default: %(default)s)", default=500)
    # parser.add_argument("-f" , "--fit"        , **argv, required=False, type=str2bool, help="whether to fit the TACF with an exponential (default: %(default)s)", default=True)
    # Window and padding
    parser.add_argument("-w"   , "--window"   , **argv, required=False, type=str     , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt"  , "--window_t" , **argv, required=False, type=int     , help="time span of the window [fs] (default: %(default)s)", default=10)
    # dielectric Spectrum
    parser.add_argument("-p" , "--plot"  , **argv, required=False , type=str    , help="output file with the dielectric constant plot (default: %(default)s)", default=None)
    parser.add_argument("-pad" , "--padding"  , **argv, required=False, type=int     , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    # Plot
    # parser.add_argument("-pir" , "--plot", **argv, required=False, type=str , help="output file for the plot (default: %(default)s)", default='IR.pdf')
    parser.add_argument("-xl", "--xlim"       , **argv, required=False, type=flist, help="x limits in frequency (default: %(default)s)", default=[30,160])
    parser.add_argument("-yl", "--ylim"       , **argv, required=False, type=flist, help="y limits in frequency (default: %(default)s)", default=[-2000,3500])
    parser.add_argument("-xs", "--x_scale"       , **argv, required=False, type=str, help="x scale (default: %(default)s)", default="log", choices=['linear','log','symlog','logit'])
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

    assert args.derivative == True, "If derivative == False there is a bug."

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    data:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    print("\tdata shape: ",data.shape)

    # #------------------#
    # if args.blocks > 0 :
    #     print("\n\tN. of blocks: ", args.blocks)
    #     print("\tBuilding blocks ... ",end="")
    #     data = reshape_into_blocks(data,args.blocks)# .T
    #     print("done")
    #     print("\tdata shape: ",data.shape)

    #------------------#
    if args.remove_mean:
        print("\n\tRemoving mean ... ",end="")
        data -= np.mean(data,axis=args.axis_corr,keepdims=True)
        print("done")

    #------------------#
    print("\n\tComputing the fluctuations ... ",end="")
    # fluctuation = np.square(data).sum(args.axis_mean)
    # fluctuation = np.atleast_2d(fluctuation) 
    fluctuation = data.std(axis=(args.axis_corr,args.axis_mean))
    print("done")
    print("\tfluctuation shape: ",fluctuation.shape)

    #------------------#
    if args.derivative:
        print("\n\tComputing the derivative ... ",end="")
        data = np.gradient(data,axis=args.axis_corr)/args.time_step
        print("done")
        print("\tdata shape: ",data.shape)

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    # if args.method == "function":
    #     autocorr = tacf(data)
    # else:
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
    if args.window != "none" :
        print("\n\tApplying the '{:s}' window ... ".format(args.window),end="")
        #Define window to be used with DCT below.
        func = getattr(np, args.window)
        window = np.zeros(autocorr.shape[args.axis_corr])
        M = int(args.window_t / args.time_step)
        window[:M] = func(2*M)[M:]
        autocorr = autocorr * window
        # window = np.atleast_2d(window)
        print("done")
        print("\twindow shape: ",window.shape)    

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
    print("\n\tComputing the frequency dependent part of the spectrum ... ", end="")
    spectrum, freq = compute_spectrum(autocorr,axis=args.axis_corr,pad=args.padding,method="rfft")
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    print("\tfreq shape: :",freq.shape)

    # #------------------#
    # print("\n\tReshaping the spectrum ... ", end="")
    # spectrum, freq = np.take(spectrum,axis=args.axis_corr,indices=np.arange(autocorr.shape[args.axis_corr]))
    # print("done")
    # print("\tspectrum shape: :",spectrum.shape)
    # print("\tfreq shape: :",freq.shape)

    #------------------#
    print("\n\tComputing the whole spectrum ... ", end="")
    # spectrum = fluctuation[:,np.newaxis] + 1.j 2*np.pi * freq * spectrum
    omega = 2*np.pi*freq
    if args.derivative:
        spectrum = fluctuation[:,np.newaxis] + 1.j*spectrum / omega
    else:
        spectrum = fluctuation[:,np.newaxis] + 1.j*omega * spectrum
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    
    # #------------------#
    # print("\n\tNormalizing the spectra ... ", end="")
    # factor   = np.max(spectrum,axis=args.axis_corr)[:, np.newaxis]
    # spectrum = np.divide(spectrum,factor)
    # std      = np.divide(std,factor)
    # print("done")
    # print("\tspectrum shape: :",spectrum.shape)    

    # assert np.allclose(np.max(spectrum,axis=args.axis_corr),1), "the spectra are not normalized"

    #------------------#
    print("\n\tComputing the average over the trajectories ... ", end="")
    std:np.ndarray = np.std(spectrum.real,axis=0) + 1.j*np.std(spectrum.imag,axis=0)
    spectrum:np.ndarray = spectrum.mean(axis=0)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"

    #------------------#
    print("\n\tComputing the frequencies ... ", end="")
    # This could require some fixing
    # Convert timestep to seconds
    dt = convert(args.time_step, "time","femtosecond", "second")
    # Compute the sampling rate in Hz
    sampling_rate = 1 / dt
    # Convert sampling rate to the desired units
    sampling_rate = convert(sampling_rate, "frequency", "hz", args.freq_unit)
    # Compute the frequency array
    freq *= sampling_rate #np.linspace(0, sampling_rate, len(spectrum))
    print("done")

    #------------------#
    print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
    tmp =  np.vstack((freq,spectrum,std)).T
    assert tmp.ndim == 2, "thsi array should have 2 dimensions"
    assert tmp.shape[1] == 3, "this array should have 3 columns"
    tmp = PhysicalTensor(tmp)
    if str(args.output).endswith("txt"):
        header = \
            f"Col 1: frequency in {args.freq_unit}\n" +\
            f"Col 2: normalized spectrum\n" +\
            f"Col 3: std (over trajectories) of the spectrum "
        tmp.to_file(file=args.output,header=header)
    else:
        tmp.to_file(file=args.output)
    
    del tmp
    print("done")


    #------------------#
    if args.plot:
        print("\tPreparing plot ... ", end="")
        fig, ax = plt.subplots(1, figsize=(6, 4))
        # y = np.linalg.norm(spectrum.mean(axis=args.axis_corr),axis=-1)
        # y /= np.max(y)
        # ax.plot(freq,y,label="raw",color="red")
        ax.plot(freq,spectrum.real, label="$\\rm \\mathcal{Re} \\chi\\left(\\omega\\right)$",color="blue") #, marker='.', markerfacecolor='blue', markersize=args.marker_size)
        ax.plot(freq,spectrum.imag, label="$\\rm \\mathcal{Im} \\chi\\left(\\omega\\right)$",color="red") #, marker='.', markerfacecolor='red', markersize=args.marker_size)

        ylow,yhigh = spectrum - std, spectrum + std
        ax.fill_between(freq,ylow.real,yhigh.real, color='gray', alpha=0.8)#, label='$\\rm \\mathcal{Re}\\left[ \\epsilon \\left(\\omega\\right)\\pm\\sigma\\left(\\omega\\right)\\right]$')
        ax.fill_between(freq,ylow.imag,yhigh.imag, color='gray', alpha=0.8)#, label='$\\rm \\mathcal{Im}\\left[ \\epsilon \\left(\\omega\\right)\\pm\\sigma\\left(\\omega\\right)\\right]$')
        # ylow,yhigh = spectrum - 2*std, spectrum + 2*std
        # ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.5, label='$\\pm2\\sigma$')
        ax.legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")
        ax.set_xlim(args.xlim[0],args.xlim[1])
        ax.set_ylim(args.ylim[0],args.ylim[1])
        # ax.set_yticks(np.arange(0,1.001,0.2))
        ax.set_xscale(args.x_scale)
        # ax.set_yscale("log")
        ax.set_xlabel("frequency [THz]")
        ax.set_ylabel("electric susceptibility [arb. units]")
        ax.grid()
        plt.tight_layout()

        print("done")

        #------------------#
        print("\tSaving plot to file '{:s}'... ".format(args.plot), end="")
        plt.savefig(args.plot)
        # plt.show()
        print("done")
    

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()