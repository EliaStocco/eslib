#!/usr/bin/env python
import numpy as np
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.formatting import esfmt
from eslib.classes.tcf import TimeAutoCorrelation
from eslib.classes.tcf import  compute_spectrum, get_freq
from eslib.tools import convert
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dielectric susceptibility (without the zero frequency contribution) from dipole time series in arbitrary units."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"          , **argv, required=True , type=str     , help="txt/npy input file")
    parser.add_argument("-o"  , "--output"         , **argv, required=False, type=str     , help="txt/npy output file (default: %(default)s)", default='dielectric.txt')
    parser.add_argument("-dt" , "--time_step"      , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-d"  , "--delta"          , **argv, required=False, type=str2bool, help="wheter to return the difference w.r.t. the zero frequency contribution (default: %(default)s)", default=True)
    parser.add_argument("-pad", "--padding"        , **argv, required=False, type=int     , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-as" , "--axis_samples"   , **argv, required=False, type=int     , help="axis corresponding to independent trajectories/samples (default: %(default)s)", default=0)
    parser.add_argument("-at" , "--axis_time"      , **argv, required=False, type=int     , help="axis corresponding to time (default: %(default)s)", default=1)
    parser.add_argument("-ac" , "--axis_components", **argv, required=False, type=int     , help="axis corresponding to dipole components (default: %(default)s)", default=2)
    parser.add_argument("-fu" , "--freq_unit"      , **argv, required=False, type=str     , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="inversecm")
    parser.add_argument("-w"  , "--window"         , **argv, required=False, type=str     , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt" , "--window_t"       , **argv, required=False, type=int     , help="time span of the window [fs] (default: %(default)s)", default=5000)
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    data:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    if data.ndim == 2:
        data = data[np.newaxis,:,:]
    print("\tdata shape: ",data.shape)
    
    #------------------#
    # This is strictly necessary 
    print("\n\tRemoving the mean ... ",end="")
    data -= np.mean(data, axis=args.axis_time,keepdims=True)
    print("done")
    
    # #------------------#
    # # norm is the average value of the time derivative of the dipole
    # print("\n\tComputing the average value of the dipole squared ... ",end="")
    # norm_dip = np.mean(data ** 2, axis=args.axis_time,keepdims=True)# *data.shape[args.axis_components]
    # # norm_dip = np.squeeze(norm_dip)
    # norm_dip = float(np.mean(np.sum(norm_dip,axis=args.axis_components)))
    # print("done")
    # print("\tdipole square: ",norm_dip)
    # data /= np.sqrt(norm_dip)
    
    # obj = TimeAutoCorrelation(data)
    # autocorr = obj.tcf(axis=args.axis_time)
    # np.mean(np.sum(autocorr,axis=2),axis=0)[0] == 1

    # #------------------#
    # print("\n\tComputing the derivative ... ",end="")
    # data = np.gradient(data,axis=args.axis_time)/args.time_step
    # print("done")
    # print("\tdata shape: ",data.shape)    

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    obj = TimeAutoCorrelation(data)
    autocorr = obj.tcf(axis=args.axis_time)
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    print("\n\tComputing the average value of the time derivative of the dipole squared ... ",end="")
    # norm is the average value of the time derivative of the dipole
    norm = np.mean(data ** 2, axis=args.axis_time,keepdims=True)# *data.shape[args.axis_components]
    norm = np.sum(norm,axis=args.axis_components,keepdims=True)
    print("done")
    print("\tnorm shape: ",norm.shape)
    
    print("\tNormalizing autocorr ... ",end="")
    autocorr /= norm
    print("done")
    print("\tautocorr shape: ",autocorr.shape)
    
    #------------------#
    print("\tComputing normalizing factor ... ",end="")
    norm = float(np.mean(np.sum(norm,axis=args.axis_components)))
    print("done")
    print("\tnormalizing factor: ",norm)

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
        print("\tdata shape: ",data.shape)

    raw_autocorr = np.copy(autocorr)  
    
    # fig,ax = plt.subplots()
    # for i in range(10):
    #     ax.plot(autocorr[i])
    # plt.show()

    #------------------#
    if args.window != "none" :
        print("\n\tApplying the '{:s}' window ... ".format(args.window),end="")
        #Define window to be used with DCT below.
        func = getattr(np, args.window)
        window = np.zeros(raw_autocorr.shape[args.axis_time])
        M = int(args.window_t / args.time_step)
        window[:M] = func(2*M)[M:]
        window /= window[0] # small correction so that the first value is 1
        # window = np.atleast_2d(window)
        print("done")
        print("\twindow shape: ",window.shape)

        autocorr = raw_autocorr * window
        
    # fig,ax = plt.subplots()
    # for i in range(10):
    #     ax.plot(autocorr[i])
    # plt.show()

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    axis_fourier = args.axis_time if args.axis_time < args.axis_components else args.axis_time - 1
    spectrum, freq = compute_spectrum(autocorr,axis=axis_fourier,pad=args.padding,method="rfft",dt=args.time_step)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    # spectrum = spectrum.real

    #------------------#
    print("\n\tComputing the whole spectrum ... ", end="")
    omega = 2*np.pi*freq
    with np.errstate(divide='ignore', invalid='ignore'):
        spectrum = 1.j * np.multiply(np.conjugate(spectrum) , omega) 
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    
    #------------------#
    if not args.delta:
        spectrum += 1
    
    #------------------#
    print("\n\tMultiplying the spectrum by the normalizing factor  ... ", end="")
    spectrum = spectrum * norm
    print("done")
    
    # #------------------#
    # print("\n\tMultiplying the spectrum by the average value of the dipole squared  ... ", end="")
    # spectrum *= norm_dip
    # print("done")

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
    freq = get_freq(dt=args.time_step, N=len(spectrum),output_units="THz")
    freq = convert(freq,'frequency','thz',args.freq_unit)
    print("done")


    #------------------#
    print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
    tmp =  np.vstack((freq,spectrum,std,err)).T
    assert tmp.ndim == 2, "thsi array should have 2 dimensions"
    # assert tmp.shape[1] == 3, "this array should have 3 columns"
    tmp = PhysicalTensor(tmp)
    if str(args.output).endswith("txt"):
        header = \
            f"Col 1: frequency in {args.freq_unit}\n" +\
            f"Col 2: dielectric susceptibility (without the zero frequency component)\n" +\
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