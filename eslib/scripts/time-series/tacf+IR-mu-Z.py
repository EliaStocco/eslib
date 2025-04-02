#!/usr/bin/env python
import numpy as np
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.tcf import TimeAutoCorrelation, compute_spectrum, get_freq
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import convert
from eslib.io_tools import pattern2sorted_files

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Infrared (IR) absorption spectrum from Born Charges and velocities time series in arbitrary units."
documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of the dipole."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"          , **argv, required=True , type=str  , help="pattern to the files with the atomic structures")
    parser.add_argument("-if" , "--input_format"   , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-z"  , "--bec"            , **argv, required=False , type=str  , help="keyword of the BEC (default: %(default)s)", default="BEC")
    parser.add_argument("-v"  , "--velocities"       , **argv, required=False , type=str  , help="keyword of the velocities (default: %(default)s)", default="velocities")
    parser.add_argument("-vu" , "--velocities_unit"  , **argv, required=False , type=str  , help="unit of the velocities (default: %(default)s)", default="atomic_unit")
    
    parser.add_argument("-o"  , "--output"         , **argv, required=False, type=str  , help="txt/npy output file (default: %(default)s)", default='IR.txt')
    parser.add_argument("-dt" , "--time_step"      , **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-s"  , "--stride"         , **argv, required=False, type=int, help="stride (default: %(default)s)", default=1)
    parser.add_argument("-T"  , "--temperature"    , **argv, required=True , type=float   , help="temperature [K]")
    
    parser.add_argument("-pad", "--padding"        , **argv, required=False, type=int  , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-as" , "--axis_samples"   , **argv, required=False, type=int  , help="axis corresponding to independent trajectories/samples (default: %(default)s)", default=0)
    parser.add_argument("-at" , "--axis_time"      , **argv, required=False, type=int  , help="axis corresponding to time (default: %(default)s)", default=1)
    parser.add_argument("-ac" , "--axis_components", **argv, required=False, type=int  , help="axis corresponding to dipole components (default: %(default)s)", default=2)
    parser.add_argument("-fu" , "--freq_unit"      , **argv, required=False, type=str  , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="inversecm")
    parser.add_argument("-w"  , "--window"         , **argv, required=False, type=str  , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt" , "--window_t"       , **argv, required=False, type=int  , help="time span of the window [fs] (default: %(default)s)", default=5000)
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    args.time_step = convert(args.stride*args.time_step,"time","femtosecond","atomic_unit")
    args.window_t = convert(args.window_t,"time","femtosecond","atomic_unit")
    
    print("\tAxes:")
    print("\t - samples:",args.axis_samples)
    print("\t - time:",args.axis_time)
    print("\t - components:",args.axis_components)    
    
    #------------------#
    print("\n\tData will be extracted from the following files:")
    files = pattern2sorted_files(args.input)
    for n,file in enumerate(files):
        print(f"\t - {n}) {file}")
    
    Zx = [None]*len(files)
    Zy = [None]*len(files)
    Zz = [None]*len(files)
    v  = [None]*len(files)
    V  = [None]*len(files)
    
    for n,file in enumerate(files):
        
        # print(f"\t - {n}) {file}")
        print(f"\n\t{n}) Reading the atomic structures from file '{args.input}' ... ", end="")
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
        print("done")
        volumes = np.asarray([ atoms.get_volume() for atoms in structures ])
        print("\n\t\tvolume [ang^3]: ",volumes.mean()," +/- ",volumes.std())
        V[n] = convert(volumes.mean(),"volume","angstrom3","atomic_unit")

        #------------------#
        print(f"\n\t\tExtracting the Born Charges using the keyword '{args.bec}' ... ", end="")
        Z = structures.get(args.bec)
        print("done")
        print("\t\tZ.shape: ",Z.shape)
        
        # print("\n\tReshaping Born Charges ... ",end="")
        # Z =
        # print("done")
        
        Zx[n] = Z[:,:,0::3] # d mu_x / d R 
        Zy[n] = Z[:,:,1::3] # d mu_x / d R
        Zz[n] = Z[:,:,2::3] # d mu_x / d R
    
        #------------------#
        print(f"\n\t\tExtracting the velocities using the keyword '{args.velocities}' ... ", end="")
        v[n] = structures.get(args.velocities)
        print("done")
        print("\t\tv.shape: ",v[n].shape)
        
        if args.velocities_unit != 'atomic_unit':
            print(f"\n\t\tConvert velocities from '{args.velocities_unit}' to 'atomic_unit'")
            v[n] = convert(v[n],"velocity",args.velocities_unit,'atomic_unit')
            
    print("\tdone")
            
    Zx = np.asarray(Zx)
    Zy = np.asarray(Zy)
    Zz = np.asarray(Zz)
    v  = np.asarray(v)
    
    #------------------#
    print("\n\tComputing the dipole time derivative ... ",end="")
    d_mu_x_dt =  np.einsum("mijk,mijk-> mi",Zx,v)
    d_mu_y_dt =  np.einsum("mijk,mijk-> mi",Zy,v)
    d_mu_z_dt =  np.einsum("mijk,mijk-> mi",Zz,v)
    data = np.zeros((len(files),d_mu_x_dt.shape[1],3))
    data[:,:,0] = d_mu_x_dt
    data[:,:,1] = d_mu_y_dt
    data[:,:,2] = d_mu_z_dt
    print("done")
    print("\tdata shape: ",data.shape)   
    
    print("\n\tHere we have:")
    print(f"\t - {data.shape[args.axis_samples]} samples") 
    print(f"\t - {data.shape[args.axis_time]} time steps") 
    print(f"\t - {data.shape[args.axis_components]} components") 
    
    
    # print("\n\tReading the input array from file '{:s}' ... ".format(args.dipole), end="")
    # args.dipole = str(args.dipole)
    # data:np.ndarray = PhysicalTensor.from_file(file=args.dipole).to_data() # e ang
    # print("done")
    # if data.ndim == 2:
    #     data = data[np.newaxis,:,:]
    # print("\tdata shape: ",data.shape)
    # data = convert(data,"electric-dipole","eang","atomic_unit")

    # #------------------#
    # print("\n\tComputing the derivative ... ",end="")
    # data = np.gradient(data,axis=args.axis_time)/args.time_step # e ang/fs
    # print("done")
    # print("\tdata shape: ",data.shape)    
    
    #------------------#
    print("\n\tRemoving the mean ... ",end="")
    data -= np.mean(data, axis=args.axis_time,keepdims=True)
    print("done")

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    obj = TimeAutoCorrelation(data)
    autocorr = obj.tcf(axis=args.axis_time) # e^2 ang^2 / fs^2
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    print("\n\tComputing the average value of the time derivative of the dipole squared ... ",end="")
    norm_der = np.mean(data ** 2, axis=args.axis_time,keepdims=True)
    print("done")
    print("\tnorm shape: ",norm_der.shape)
    
    print("\tNormalizing autocorr ... ",end="")
    autocorr /= norm_der
    print("done")
    print("\tautocorr shape: ",autocorr.shape)
    
    #------------------#
    print("\tComputing normalizing factor ... ",end="")
    norm_der = float(np.mean(np.sum(norm_der,axis=args.axis_components)))
    print("done")
    print("\tnormalizing factor: ",norm_der)

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

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    axis_fourier = args.axis_time if args.axis_time < args.axis_components else args.axis_time - 1
    spectrum, freq = compute_spectrum(autocorr,axis=axis_fourier,pad=args.padding,method="rfft",dt=args.time_step) # e^2 ang^2 / fs
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    
    #------------------#
    print("\n\tMultiplying the spectrum by the normalizing factor  ... ", end="")
    spectrum = spectrum.real*norm_der
    print("done")
    
    #------------------#
    # print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    # atoms = read(args.input,format=args.input_format,index=0)
    # print("done")
    # volume = atoms.get_volume()
    # print(f"\tvolume: {volume:.2f} ang3")
    # volume = convert(volume,"volume","angstrom3","atomic_unit")
    args.temperature = convert(args.temperature,"temperature","kelvin","atomic_unit")
    epsilon = 1/(4*np.pi)
    kB = 1
    c = convert(299792458,"velocity","m/s","atomic_unit")
    factor = 3*c*np.asarray(V)*args.temperature*kB*epsilon/np.pi
    spectrum /= factor[:,None]
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