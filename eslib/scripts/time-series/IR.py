#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool, flist
from eslib.formatting import esfmt
from eslib.classes.tcf import compute_spectrum, get_freq

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Infra Red (IR) absorption spectrum."
documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of dipole."

alpha = 0.5

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"      , **argv, required=True , type=str     , help="txt/npy input file with the dipole TACF")
    parser.add_argument("-o"  , "--output"     , **argv, required=False, type=str     , help="output file with the Infrared spectrum (default: %(default)s)", default="IR.txt")
    parser.add_argument("-dt" , "--time_step"  , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-d"  , "--derivative" , **argv, required=False, type=str2bool, help="whether the TACF has been computed from time derivatives or dipoles (default: %(default)s)", default=True)
    parser.add_argument("-a"  , "--axis"       , **argv, required=False, type=int     , help="axis along compute the FFT (default: %(default)s)", default=1)
    parser.add_argument("-pad", "--padding"    , **argv, required=False, type=int     , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-n"  , "--normalize"  , **argv, required=False, type=str2bool, help="whether to normalize the spectrum (default: %(default)s)", default=False)
    parser.add_argument("-p"  , "--plot"       , **argv, required=False, type=str     , help="output file for the plot (default: %(default)s)", default='IR.pdf')
    # parser.add_argument("-mf" , "--max_freq"   , **argv, required=False, type=float   , help="max frequency in IR plot [THz] (default: %(default)s)", default=500)
    parser.add_argument("-fu" , "--freq_unit"  , **argv, required=False, type=str     , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="THz")
    parser.add_argument("-xl", "--xlim"        , **argv, required=False, type=flist   , help="x limits in frequency (default: %(default)s)", default=[0,None])
    parser.add_argument("-yl", "--ylim"        , **argv, required=False, type=flist   , help="y limits in frequency (default: %(default)s)", default=[0,None])
    parser.add_argument("-ms" , "--marker_size", **argv, required=False, type=float   , help="marker size (default: %(default)s)", default=0)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the autocorrelation from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    autocorr:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if autocorr.ndim == 1:
        print("\tReshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)  
        print("done")
        print("\tdata shape: ",autocorr.shape)

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    spectrum, freq = compute_spectrum(autocorr,axis=args.axis,pad=args.padding,method="rfft",dt=args.time_step)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    spectrum = spectrum.real

    #------------------#
    if not args.derivative:
        print("\n\tThe autocorr did not come from dipole derivatives: multiplying the spectrum for omega^2 ... ", end="")
        spectrum = np.multiply(spectrum,freq**2)
        print("done")
        print("\tspectrum shape: :",spectrum.shape)

    #------------------#
    if args.normalize:
        print("\n\tNormalizing the spectra ... ", end="")
        factor   = np.max(spectrum,axis=args.axis)[:, np.newaxis]
        spectrum = np.divide(spectrum,factor)
        # std      = np.divide(std,factor)
        print("done")
        print("\tspectrum shape: :",spectrum.shape)    

        assert np.allclose(np.max(spectrum,axis=args.axis),1), "the spectra are not normalized"

    #------------------#
    print("\n\tComputing the average over the trajectories ... ", end="")
    N = spectrum.shape[0]
    std:np.ndarray = spectrum.std(axis=0)
    spectrum:np.ndarray = spectrum.mean(axis=0)
    err:np.ndarray = std / np.sqrt(N)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"

    #------------------#
    print("\n\tComputing the frequencies ... ", end="")
    freq = get_freq(dt=args.time_step, N=len(spectrum),output_units=args.freq_unit)
    print("done")

    print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
    tmp =  np.vstack((freq,spectrum,std,err)).T
    assert tmp.ndim == 2, "thsi array should have 2 dimensions"
    assert tmp.shape[1] == 4, "this array should have 4 columns"
    tmp = PhysicalTensor(tmp)
    if str(args.output).endswith("txt"):
        normalized = "normalized " if args.normalize else ""
        header = \
            f"Col 1: frequency in {args.freq_unit}\n" +\
            f"Col 2: {normalized}spectrum\n" +\
            f"Col 3: std (over trajectories) of the spectrum\n" +\
            f"Col 4: error of the spectrum (std/sqrt(N), with N = n. of trajectories)"
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
        ax.plot(freq,spectrum, label="$\\rm S\\left(\\omega\\right)$",color="blue", marker='.', markerfacecolor='blue', markersize=args.marker_size)
        ylow,yhigh = spectrum - err, spectrum + err
        ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.8)

        ax.legend(loc="upper left",facecolor='white', framealpha=1,edgecolor="black")
        # ax.set_xlim(0,args.max_freq)
        # ax.set_ylim(0,None)
        ax.set_xlim(args.xlim[0],args.xlim[1])
        ax.set_ylim(args.ylim[0],args.ylim[1])
        if args.normalize:
            ax.set_yticks(np.arange(0,1.001,0.2))
        ax.set_ylabel("spectrum [arb. units]")
        ax.set_xlabel(f"frequency [{args.freq_unit}]")        
        ax.grid()
        plt.tight_layout()
        print("done")

        #------------------#
        print("\tSaving plot to file '{:s}'... ".format(args.plot), end="")
        plt.savefig(args.plot)
        print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()