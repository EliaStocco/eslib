#!/usr/bin/env python
import numpy as np
from ase.io import read
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.tcf import compute_spectrum, get_freq, autocorrelate
from eslib.formatting import esfmt, eslog
from eslib.tools import convert
from eslib.mathematics import centered_window
from eslib.input import itype, str2bool
from eslib.io_tools import numpy_take
from scipy import signal
from eslib.mathematics import mean_std_err

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Infrared (IR) absorption spectrum from dipole time series in arbitrary units."
documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of dipole."

PARALLEL = False # better for few samples

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-d"  , "--dipole"          , **argv, required=True , type=str  , help="txt/npy input file")
    parser.add_argument("-id" , "--is_derivative"   , **argv, required=False, type=str2bool, help="whether the input is the time derivative of the dipole [or the dipole itself] (default: %(default)s)", default=False)
    parser.add_argument("-iu" , "--input_unit"      , **argv, required=False, type=str  , help="input unit (default: %(default)s)", default=None)
    parser.add_argument("-o"  , "--output"          , **argv, required=False, type=str  , help="txt output file with the spectrum (default: %(default)s)", default='IR.txt')
    parser.add_argument("-oac", "--output_autocorr" , **argv, required=False, type=str  , help="txt output file with autocorrelation function (default: %(default)s)", default=None)
    parser.add_argument("-dt" , "--time_step"       , **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-T"  , "--temperature"     , **argv, required=True , type=float, help="temperature [K]")
    parser.add_argument("-i"  , "--input"           , **argv, required=True , type=str  , help="file with the atomic structure")
    parser.add_argument("-if" , "--input_format"    , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-pad", "--padding"         , **argv, required=False, type=int  , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-n"  , "--index"           , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
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
    
    #------------------#
    print("\tAxes:")
    print("\t - samples:",args.axis_samples)
    print("\t - time:",args.axis_time)
    print("\t - components:",args.axis_components)    
    
    #------------------#
    dt = args.time_step
    args.time_step = convert(args.time_step,"time","femtosecond","atomic_unit")
    args.window_t = convert(args.window_t,"time","femtosecond","atomic_unit")
    args.dipole = str(args.dipole)

    #------------------#
    with eslog(f"\nReading the input array from file '{args.dipole}'"):
        data:np.ndarray = PhysicalTensor.from_file(file=args.dipole).to_data() # e ang
    if data.ndim == 2:
        data = data[np.newaxis,:,:]
    print("\t data shape: ",data.shape)
    
    if args.index is not None and args.index != ":":
        print("\n\t Resampling data ... ", end="")
        data = numpy_take(data, args.index, axis=args.axis_time)
        print("done")
        print("\t data shape: ",data.shape)
        
    if not args.is_derivative:
        
        if args.input_unit is None:
            args.input_unit = "eang"
        
        print(f"\n\t Converting data from '{args.input_unit}' to 'atomic_unit' ... ",end="")
        data = convert(data,"electric-dipole",args.input_unit,"atomic_unit")
        print("done")
        
        #------------------#
        print("\n\t Computing the time derivative ... ",end="")
        data = np.gradient(data,axis=args.axis_time)/args.time_step # e ang/fs
        print("done")
        print("\t data shape: ",data.shape)   
        
    else:
        
        assert args.input_unit == None, "No unit conversion suppported for the time derivative of the dipole." 
        
    #------------------#
    print("\n\t Removing the mean ... ",end="")
    data -= np.mean(data, axis=args.axis_time,keepdims=True)
    print("done")

    #------------------#
    with eslog("\nComputing the autocorrelation function"):
        autocorr = autocorrelate(data,axis=args.axis_time,use_parallel=PARALLEL) # e^2 ang^2 / fs^2
    print("\t autocorr shape: ",autocorr.shape)
    
    #------------------#
    print("\n\t Computing the average value of the time derivative of the dipole squared ... ",end="")
    norm_der = np.mean(data ** 2, axis=args.axis_time,keepdims=True)
    print("done")
    print("\t norm shape: ",norm_der.shape)
    
    print("\n\t Normalizing autocorr ... ",end="")
    assert np.allclose(autocorr[:,int(autocorr.shape[1]/2),:],norm_der[:,0,:]), "the normalization factor is not correct"
    autocorr /= norm_der
    assert np.allclose(autocorr[:,int(autocorr.shape[1]/2),:],1), "the normalization factor is not correct"
    print("done")
    print("\t autocorr shape: ",autocorr.shape)
    
    #------------------#
    print("\n\t Printing <mu_i^2>:")
    tmp = np.moveaxis(norm_der, [args.axis_samples,args.axis_components], [0,1])
    assert tmp.shape[2] == 1, "the shape of the array is not correct"
    tmp = tmp[:,:,0]
    print("\t ---------------------------------------------------")
    print("\t | {:>8} | {:>10} | {:>10} | {:>10} |".format("sample","x","y","z"))
    print("\t ---------------------------------------------------")
    for i in range(tmp.shape[0]):
        print(f"\t | {i:>8} | {tmp[i,0]:10.4e} | {tmp[i,1]:10.4e} | {tmp[i,2]:10.4e} |")
    print("\t ---------------------------------------------------")
    
    #------------------#
    print("\n\t Computing normalizing factor ... ",end="")
    norm_std = float(np.std(np.sum(norm_der,axis=args.axis_components)))
    norm_der = float(np.mean(np.sum(norm_der,axis=args.axis_components)))
    print("done")
    print(f"\t     norm. factor: {norm_der:.2e} [e^2ang^2]")
    print(f"\t norm. factor std: {norm_std:.2e} [e^2ang^2]")
    print(f"\t norm. factor std: {100*norm_std/norm_der:.2f} [%] (this should be small)")

    #------------------#
    if args.axis_components is not None:
        print("\n\t Computing the sum along the axis {:d} ... ".format(args.axis_components),end="")
        autocorr = np.sum(autocorr,axis=args.axis_components)# /np.sqrt(autocorr.shape[args.axis_components])
        print("done")
        print("\t autocorr shape: ",autocorr.shape)

    #------------------#
    if autocorr.ndim == 1:
        print("\t Reshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)# .T
        print("done")
        print("\t data shape: ",data.shape)
    
    raw_autocorr = np.copy(autocorr)  

    #------------------#
    if args.window != "none" :
        print("\n\t Applying the '{:s}' window ... ".format(args.window),end="")
        M = int(args.window_t / args.time_step)
        window = centered_window(args.window,raw_autocorr.shape[args.axis_time],M)
        print("done")
        print("\t window shape: ",window.shape)
        autocorr = raw_autocorr * window
                
    #------------------#
    print("\n\t Computing the spectrum ... ", end="")
    # axis_fourier = args.axis_time if args.axis_time < args.axis_components else args.axis_time - 1
    spectrum, freq = compute_spectrum(autocorr,axis=args.axis_time,pad=args.padding,dt=args.time_step,shift=True) # e^2 ang^2 / fs
    print("done")
    print("\t spectrum shape: :",spectrum.shape)
    
    #------------------#
    print("\n\t Multiplying the spectrum by the normalizing factor  ... ", end="")
    spectrum = spectrum.real*norm_der
    print("done")
    
    #------------------#
    print("\t Reading atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=0)
    print("done")
    volume = atoms.get_volume()
    print(f"\t volume: {volume:.2f} ang3")
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
    print("\n\t Computing the average over the trajectories ... ", end="")
    N = spectrum.shape[args.axis_samples]
    std:np.ndarray      = np.std (spectrum,axis=args.axis_samples)
    spectrum:np.ndarray = np.mean(spectrum,axis=args.axis_samples)
    err:np.ndarray      = std/np.sqrt(N-1)
    print("done")
    print("\t spectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"
    
    # print(max(spectrum.real))

    #------------------#
    print("\n\t Computing the frequencies ... ", end="")
    freq = get_freq(dt=args.time_step, N=len(spectrum),input_units="atomic_unit",output_units=args.freq_unit)
    # freq = convert(freq,'frequency','thz',args.freq_unit)
    print("done")
    print("\t max freq: ",freq[-1],f" {args.freq_unit}")

    #------------------#
    print("\n\t Saving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
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
    
    #------------------#
    if args.output_autocorr is not None:
        print("\n\t Saving the autocorrelation and time to file '{:s}' ... ".format(args.output_autocorr), end="")
        
        # func = extend2NDarray(np.fft.ifftshift)
        # autocorr = func(autocorr,axis=args.axis_time)   
        N = data.shape[args.axis_time]
        lags = signal.correlation_lags(N,N, mode="full")*dt
        mean,std,err = mean_std_err(autocorr,axis=args.axis_samples)
        
        tmp =  np.vstack((lags,mean,std,err)).T
        assert tmp.ndim == 2, "this array should have 2 dimensions"
        tmp = PhysicalTensor(tmp)
        header = \
            f"Col 1: time/lag in femtosecond\n" +\
            f"Col 2: autocorrelation function (normalized)\n" +\
            f"Col 3: std (over trajectories) of the autocorrelation\n"+\
            f"Col 4: error of the autocorrelation (std/sqrt(N-1), with N = n. of trajectories)"
        tmp.to_file(file=args.output_autocorr,header=header)
        print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()