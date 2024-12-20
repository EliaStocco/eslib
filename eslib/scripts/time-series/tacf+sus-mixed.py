#!/usr/bin/env python
import numpy as np
from ase.io import read
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.tcf import TimeCorrelation, compute_spectrum, get_freq
from eslib.formatting import esfmt
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dielectric susceptibility from dipole time series."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-d"  , "--dipole"         , **argv, required=True , type=str     , help="txt/npy input file with the dipoles [eang]")
    parser.add_argument("-o"  , "--output"         , **argv, required=False, type=str     , help="txt/npy output file (default: %(default)s)", default='sus.txt')
    parser.add_argument("-dt" , "--time_step"      , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-T"  , "--temperature"    , **argv, required=True , type=float   , help="temperature [K]")
    parser.add_argument("-i"  , "--input"          , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-if" , "--input_format"   , **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-pad", "--padding"        , **argv, required=False, type=int     , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-as" , "--axis_samples"   , **argv, required=False, type=int     , help="axis corresponding to independent trajectories/samples (default: %(default)s)", default=0)
    parser.add_argument("-at" , "--axis_time"      , **argv, required=False, type=int     , help="axis corresponding to time (default: %(default)s)", default=1)
    parser.add_argument("-ac" , "--axis_components", **argv, required=False, type=int     , help="axis corresponding to dipole components (default: %(default)s)", default=2)
    parser.add_argument("-fu" , "--freq_unit"      , **argv, required=False, type=str     , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="inversecm")
    parser.add_argument("-w"  , "--window"         , **argv, required=False, type=str     , help="window type (default: %(default)s)", default='hanning', choices=['none','bartlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt" , "--window_t"       , **argv, required=False, type=int     , help="time span of the window [fs] (default: %(default)s)", default=5000)
    parser.add_argument("-B"  , "--beta"           , **argv, required=False, type=float   , help="beta value for the Kaiser window (default: %(default)s)", default=1)
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    args.time_step = convert(args.time_step,"time","femtosecond","atomic_unit")
    args.window_t = convert(args.window_t,"time","femtosecond","atomic_unit")

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.dipole), end="")
    args.dipole = str(args.dipole)
    dipole:np.ndarray = PhysicalTensor.from_file(file=args.dipole).to_data() # e ang
    print("done")
    if dipole.ndim == 2:
        dipole = dipole[np.newaxis,:,:]
    print("\tdipole shape: ",dipole.shape)
    dipole = convert(dipole,"electric-dipole","eang","atomic_unit")
    
    # #------------------#
    # print("\n\tRemoving the mean ... ",end="")
    # dipole -= np.mean(dipole, axis=args.axis_time,keepdims=True)
    # print("done")
    
    #------------------#
    # norm_der is the average value of the time derivative of the dipole
    print("\n\tComputing the average value of the dipole squared ... ",end="")
    norm_dip = np.mean(dipole ** 2, axis=args.axis_time,keepdims=True)# *data.shape[args.axis_components]
    # norm_dip = np.squeeze(norm_dip)
    norm_dip = float(np.mean(np.sum(norm_dip,axis=args.axis_components)))
    print("done")
    print("\tdipole square: ",norm_dip)
    dipole /= np.sqrt(norm_dip)   
    
    #------------------#
    print("\n\tComputing the derivative ... ",end="")
    time_der = np.gradient(dipole,axis=args.axis_time)/args.time_step # e ang / fs
    print("done")
    print("\tdipole derivative shape: ",dipole.shape) 

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    obj = TimeCorrelation(dipole,time_der)
    autocorr = obj.tcf(axis=args.axis_time) # e^2 ang^2 / fs
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if args.axis_components is not None:
        print("\n\tComputing the sum along the axis {:d} ... ".format(args.axis_components),end="")
        autocorr = np.sum(autocorr,axis=args.axis_components)
        print("done")
        print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if autocorr.ndim == 1:
        print("\tReshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)# .T
        
        print("done")
        print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if args.window != "none" :
        print("\n\tApplying the '{:s}' window ... ".format(args.window),end="")
        #Define window to be used with DCT below.
        func = getattr(np, args.window)
        window = np.zeros(autocorr.shape[args.axis_time])
        M = int(args.window_t / args.time_step)
        if "kaiser" in args.window:
            W = func(2*M,args.beta)[M:]
        else:
            W = func(2*M)[M:]
        if M < len(window):
            window[:M] = W
        else:
            window = W[:len(window)]
        window /= window[0] # small correction so that the first value is 1
        # window = np.atleast_2d(window)
        print("done")
        print("\twindow shape: ",window.shape)

        autocorr = autocorr*window
        # tmp = tmp - np.mean(tmp, axis=args.axis_samples,keepdims=True) + np.mean(autocorr, axis=args.axis_time,keepdims=True)
        # autocorr = tmp/np.take(tmp,indices=0,axis=args.axis_time)

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    axis_fourier = args.axis_time if args.axis_time < args.axis_components else args.axis_time - 1
    spectrum, freq = compute_spectrum(autocorr,axis=axis_fourier,pad=args.padding,method="rfft",dt=args.time_step) # e^2 ang^2
    spectrum = -spectrum.real + 1j*spectrum.imag
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    
    #------------------#
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=0)
    print("done")
    volume = atoms.get_volume()
    print(f"\tvolume: {volume:.2f} ang3")
    volume = convert(volume,"volume","angstrom3","atomic_unit")
    args.temperature = convert(args.temperature,"temperature","kelvin","atomic_unit")
    epsilon = 1/(4*np.pi)
    kB = 1
    factor = 3*volume*args.temperature*epsilon * kB/np.pi
    spectrum /= factor
    
    #------------------#
    print("\n\tMultiplying the spectrum by the average value of the dipole squared  ... ", end="")
    spectrum *= norm_dip
    print("done")
        
   #------------------#
    print("\n\tComputing the average over the trajectories ... ", end="")
    N = spectrum.shape[args.axis_samples]
    std:np.ndarray = np.std(spectrum.real,axis=args.axis_samples) + 1.j*np.std(spectrum.imag,axis=args.axis_samples)
    spectrum:np.ndarray = spectrum.mean(axis=args.axis_samples)
    err:np.ndarray = std/np.sqrt(N-1)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"

    #------------------#
    print("\n\tComputing the frequencies ... ", end="")
    freq = get_freq(dt=args.time_step, N=len(spectrum),input_units="atomic_unit",output_units=args.freq_unit)
    # freq = convert(freq,'frequency','thz',args.freq_unit)
    print("done")


    #------------------#
    print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
    tmp =  np.vstack((freq,spectrum,std,err)).T
    assert tmp.ndim == 2, "this array should have 2 dimensions"
    tmp = PhysicalTensor(tmp)
    if str(args.output).endswith("txt"):
        header = \
            f"Col 1: frequency in {args.freq_unit}\n" +\
            f"Col 2: dielectric susceptibility\n" +\
            f"Col 3: std (over trajectories) of the spectrum\n"+\
            f"Col 4: error of the spectrum (std/sqrt(N-1), with N = n. of trajectories)"
        tmp.to_file(file=args.output,header=header)
    else:
        tmp.to_file(file=args.output)
    
    del tmp
    print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()