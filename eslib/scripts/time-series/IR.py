#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from eslib.formatting import esfmt, message
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.spectrum import Spectrum

matplotlib.use('QtAgg')
plt.ion()

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Infra Red (IR) spectrum given a (set of) dipole time series."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str  , help="txt/npy input file")
    parser.add_argument("-dt", "--time_step"       , **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-w", "--window"        , **argv, required=False, type=str, help="window type (default: %(default)s)", default='hanning', choices=['barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wN", "--window_N"        , **argv, required=False, type=int, help="lenght of the window (default: %(default)s)", default=500)
    parser.add_argument("-fmax", "--max_freq"       , **argv, required=False, type=float, help="maximum frequency [THz] (default: %(default)s)", default=125)
    parser.add_argument("-o" , "--output"          , **argv, required=False , type=str  , help="txt/npy output file for the spectrum (default: %(default)s)", default='spectrum.npy')
    # parser.add_argument("-od", "--output_dataframe", **argv, required=False , type=str  , help="csv output file for the dataframe (default: %(default)s)", default='spectrum.csv')
    parser.add_argument("-p" , "--plot"            , **argv, required=False, type=str  , help="output file for the plot (default: %(default)s)", default="spectrum.pdf")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\n\t{:s}: the input array will be considered to have the following dimension:".format(message))
    print("\t0: trajectory")
    print("\t1: time")
    print("\t2: dipole component (x,y,z)")
    print()

    #------------------#
    print("\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    dipoles:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    assert dipoles.ndim == 3
    assert dipoles.shape[2] == 3, "the dipoles do not have 3 components"
    # new_dims = ["trajectory","time","component"]
    # dipoles = dipoles.rename({dim: new_dim for dim, new_dim in zip(dipoles.dims, new_dims)})
    print("\tdata shape: :",dipoles.shape)

    # #------------------#
    axis = 1 # dipoles.dims.index('time')

    #------------------#
    print("\tremoving mean ... ",end="")
    dipoles -= dipoles.mean(axis=axis,keepdims=True)
    print("done")

    #------------------#
    obj = Spectrum(dipoles)

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    autocorr = obj.tcf(axis=axis)
    print("done")
    print("\tautocorr shape: :",autocorr.shape)

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    spectrum = obj.spectrum(axis=axis,autocorr=autocorr,M=args.window_N,method=args.window)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    #------------------#
    print("\n\tComputing the average over the cartesian components ... ", end="")
    spectrum:np.ndarray = np.linalg.norm(spectrum,axis=-1)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    #------------------#
    print("\n\tComputing the average over the trajectories ... ", end="")
    std:np.ndarray = spectrum.std(axis=0)
    spectrum:np.ndarray = spectrum.mean(axis=0)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    #------------------#
    print("\n\tNormalizing the spectrum ... ", end="")
    factor   = np.max(spectrum)
    spectrum = np.divide(spectrum,factor)
    std      = np.divide(std,factor)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)    

    assert spectrum.ndim == 1

    #------------------#
    print("\n\tSaving the spectrum to file '{:s}' ... ".format(args.output), end="")
    tmp = PhysicalTensor(spectrum)
    tmp.to_file(file=args.output)
    del tmp
    print("done")

    #------------------#
    print("\n\tComputing the frequencies ... ", end="")
    freq = np.fft.rfftfreq(n=dipoles.shape[axis],d=args.time_step) * 1000
    print("done")


    #------------------#
    print("\tPreparing plot ... ", end="")
    fig, ax = plt.subplots(1, figsize=(10, 4))
    y = np.linalg.norm(obj._raw_spectrum.mean(axis=0),axis=-1)
    y /= np.max(y)
    # ax.plot(freq,y,label="raw",color="red")
    ax.plot(freq,spectrum,label="spectrum",color="blue")
    ylow,yhigh = spectrum - std, spectrum + std
    ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.8, label='$\\pm\\sigma$')
    # ylow,yhigh = spectrum - 2*std, spectrum + 2*std
    # ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.5, label='$\\pm2\\sigma$')
    ax.legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")
    ax.set_xlim(0,args.max_freq)
    ax.set_xlabel("freq. [THz]")
    ax.set_ylabel("norm. spectrum")
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