#!/usr/bin/env python
import numpy as np
from ase.io import read
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.tcf import TimeAutoCorrelation, compute_spectrum, get_freq, autocorrelate
from eslib.formatting import esfmt
from eslib.tools import convert
from scipy import interpolate
# from scipy.signal import butter, filtfilt, correlate
import scipy

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Infrared (IR) absorption spectrum from dipole time series in arbitrary units."
documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of dipole."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-d"  , "--dipole"          , **argv, required=True , type=str  , help="txt/npy input file")
    parser.add_argument("-o"  , "--output"          , **argv, required=False, type=str  , help="txt/npy output file (default: %(default)s)", default='IR.txt')
    parser.add_argument("-dt" , "--time_step"       , **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-T"  , "--temperature"     , **argv, required=True , type=float   , help="temperature [K]")
    parser.add_argument("-i"  , "--input"           , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-if" , "--input_format"    , **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-pad", "--padding"         , **argv, required=False, type=int  , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    # parser.add_argument("-int", "--interpolate"     , **argv, required=False, type=int  , help="interpolation sampling multiplier (default: %(default)s)", default=1)
    # parser.add_argument("-inm", "--interpolate_mode", **argv, required=False, type=int  , help="interpolation mode (default: %(default)s)", default="none",choices=['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next','none'])
    parser.add_argument("-as" , "--axis_samples"    , **argv, required=False, type=int  , help="axis corresponding to independent trajectories/samples (default: %(default)s)", default=0)
    parser.add_argument("-at" , "--axis_time"       , **argv, required=False, type=int  , help="axis corresponding to time (default: %(default)s)", default=1)
    parser.add_argument("-ac" , "--axis_components" , **argv, required=False, type=int  , help="axis corresponding to dipole components (default: %(default)s)", default=2)
    parser.add_argument("-fu" , "--freq_unit"       , **argv, required=False, type=str  , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="inversecm")
    parser.add_argument("-w"  , "--window"          , **argv, required=False, type=str  , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt" , "--window_t"        , **argv, required=False, type=int  , help="time span of the window [fs] (default: %(default)s)", default=5000)
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    args.time_step = convert(args.time_step,"time","femtosecond","atomic_unit")
    args.window_t = convert(args.window_t,"time","femtosecond","atomic_unit")

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.dipole), end="")
    args.dipole = str(args.dipole)
    data:np.ndarray = PhysicalTensor.from_file(file=args.dipole).to_data() # e ang
    print("done")
    if data.ndim == 2:
        data = data[np.newaxis,:,:]
    print("\tdata shape: ",data.shape)
    data = convert(data,"electric-dipole","eang","atomic_unit")

    #------------------#
    print("\n\tComputing the derivative ... ",end="")
    data = np.gradient(data,axis=args.axis_time)/args.time_step # e ang/fs
    print("done")
    print("\tdata shape: ",data.shape)    
    
    #------------------#
    print("\n\tRemoving the mean ... ",end="")
    data -= np.mean(data, axis=args.axis_time,keepdims=True)
    print("done")
    
    # #------------------#
    # print("\n\tApplying high-pass filter ... ", end="")
    # cutoff_frequency = 1e-4  # Define the cutoff frequency in normalized units (0 to 0.5)
    # b, a = butter(N=4, Wn=cutoff_frequency, btype='high', fs=1/args.time_step)
    # data = filtfilt(b, a, data, axis=args.axis_time)
    # print("done")
    # print("\tdata shape after filtering: ", data.shape)
    
    #------------------#
    # if args.interpolate_mode != "none":
    # args.interpolate = 1
    # t_original = np.linspace(0,1, data.shape[args.axis_time])
    # t_new = np.linspace(0,1, data.shape[args.axis_time]*args.interpolate)
    # interp_func = interpolate.interp1d(t_original, data, kind=args.interpolate_mode,axis=args.axis_time)  # cubic spline interpolation
    # data = interp_func(t_new)
    # args.time_step /= args.interpolate

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    obj = TimeAutoCorrelation(data)
    autocorr = obj.tcf(axis=args.axis_time,mode="full") # e^2 ang^2 / fs^2
    
    
    
    print("done")
    # print("\tautocorr shape: ",autocorr.shape)
    
    test = autocorrelate(data,axis=args.axis_time)
    
    import matplotlib.pyplot as plt
    
    plt.plot(test[0,int(test.shape[1]/2):,0],label="test")
    plt.plot(autocorr[0,:,0],label="autocorr")
    plt.legend()
    plt.show()
    
    n_data = data.shape[1]
    norm = n_data - np.abs(np.arange(1-n_data, n_data), dtype=float)
    cf = correlate(data[0,:,0], data[0,:,0], method="auto")[::-1] / norm

    #------------------#
    if args.axis_components is not None:
        print("\n\tComputing the sum along the axis {:d} ... ".format(args.axis_components),end="")
        autocorr = np.sum(autocorr,axis=args.axis_components)
        print("done")
        print("\tautocorr shape: ",autocorr.shape)
        
    # #------------------#
    # print("\n\tComputing the average value of the time derivative of the dipole squared ... ",end="")
    # norm_der = np.mean(data ** 2, axis=args.axis_time,keepdims=True)
    # print("done")
    # print("\tnorm shape: ",norm_der.shape)
    
    # print("\tNormalizing autocorr ... ",end="")
    # autocorr /= autocorr[:,0][:,np.newaxis]
    # print("done")
    # print("\tautocorr shape: ",autocorr.shape)
    
    # #------------------#
    # print("\tComputing normalizing factor ... ",end="")
    # norm_der = float(np.mean(np.sum(norm_der,axis=args.axis_components)))
    # print("done")
    # print("\tnormalizing factor: ",norm_der)
    
    # autocorr = cf

    #------------------#
    if autocorr.ndim == 1:
        print("\tReshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)# .T
        print("done")
        print("\tdata shape: ",data.shape)
    raw_autocorr = np.copy(autocorr)  

    #------------------#
    if args.window != "none" :
        print("\n\tApplying the '{:s}' window ... ".format(args.window),end="")
        func = getattr(np, args.window)
        window = np.zeros(raw_autocorr.shape[args.axis_time])
        M = int(args.window_t / args.time_step)
        window[:M] = func(2*M)[M:]
        window /= window[0] # small correction so that the first value is 1
        print("done")
        print("\twindow shape: ",window.shape)
        autocorr = raw_autocorr * window
        
        # #------------------#
        # print("\n\tPerforming inverse Fourier transform ... ", end="")
        # autocorr_time_domain = np.fft.irfft(autocorr, axis=args.axis_time)
        # print("done")
        # print("\tautocorr_time_domain shape: ", autocorr_time_domain.shape)

        # #------------------#
        # print("\n\tRemoving the mean from the time-domain data ... ", end="")
        # autocorr_time_domain -= np.mean(np.sqrt(autocorr_time_domain), axis=args.axis_time, keepdims=True)
        # print("done")

        # #------------------#
        # print("\n\tTransforming the data back to the frequency domain ... ", end="")
        # autocorr = np.fft.rfft(autocorr_time_domain, axis=args.axis_time)
        # print("done")
        # print("\tautocorr shape: ", autocorr.shape)

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    axis_fourier = args.axis_time if args.axis_time < args.axis_components else args.axis_time - 1
    spectrum, freq = compute_spectrum(autocorr,axis=axis_fourier,pad=args.padding,method="rfft",dt=args.time_step) # e^2 ang^2 / fs
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    
    #------------------#
    print("\n\tMultiplying the spectrum by the normalizing factor  ... ", end="")
    spectrum = spectrum.real# *norm_der
    print("done")
    
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
    c = convert(299792458,"velocity","m/s","atomic_unit")
    factor = 3*c*volume*args.temperature*kB*epsilon/np.pi
    spectrum /= factor
    spectrum /= convert(1,"length","atomic_unit","centimeter")
    
    # print(max(spectrum.real))

   #------------------#
    print("\n\tComputing the average over the trajectories ... ", end="")
    N = spectrum.shape[args.axis_samples]
    std:np.ndarray      = np.std (spectrum,axis=args.axis_samples)
    spectrum:np.ndarray = np.mean(spectrum,axis=args.axis_samples)
    err:np.ndarray      = std/np.sqrt(N-1)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"
    
    # print(max(spectrum.real))

    #------------------#
    print("\n\tComputing the frequencies ... ", end="")
    freq = get_freq(dt=args.time_step, N=len(spectrum),input_units="atomic_unit",output_units=args.freq_unit)
    # freq = convert(freq,'frequency','thz',args.freq_unit)
    print("done")
    print("\tmax freq: ",freq[-1],f" {args.freq_unit}")

    #------------------#
    print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
    tmp =  np.vstack((freq,spectrum,std,err)).T
    assert tmp.ndim == 2, "this array should have 2 dimensions"
    tmp = PhysicalTensor(tmp)
    if str(args.output).endswith("txt"):
        header = \
            f"Col 1: frequency in {args.freq_unit}\n" +\
            f"Col 2: infrared absorption spectrum in inversecm\n" +\
            f"Col 3: std (over trajectories) of the spectrum\n"+\
            f"Col 4: error of the spectrum (std/sqrt(N-1), with N = n. of trajectories)"
        tmp.to_file(file=args.output,header=header)
    else:
        tmp.to_file(file=args.output)
    print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()