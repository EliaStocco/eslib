#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool, flist
from eslib.formatting import esfmt
from eslib.classes.tcf import  compute_spectrum, get_freq
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Compute the frequency dependent electric susceptibility (defined as dielectric constant -1) from a dipole time series."

documentation = "It's better that in `tacf.py` you used:\n - `--derivative true`\n - `--normalize anti-derivative-all`"

alpha = 0.5

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"       , **argv, required=True , type=str     , help="txt/npy input file with the dipole TACF")
    parser.add_argument("-n" , "--normalize" , **argv, required=False, type=str2bool, help="whether to normalize the autocorrelation (default: %(default)s)", default=False)
    parser.add_argument("-id" , "--input_dipole", **argv, required=False, type=str     , help="txt/npy input file with the dipoles (provide only if TACF was not normalized)", default=None)
    parser.add_argument("-in" , "--input_norm", **argv, required=False, type=str       , help="txt/npy input file with the normalizing constant (default: %(default)s)", default=None)
    parser.add_argument("-rz"  , "--remove_zero", **argv, required=False, type=str2bool, help="remove zero frequency contribution (default: %(default)s)", default=True)
    parser.add_argument("-o"  , "--output"      , **argv, required=False, type=str     , help="output file with the freq. dep. dielectric constant (default: %(default)s)", default="dielectric.txt")
    parser.add_argument("-dt" , "--time_step"   , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-d"  , "--derivative"  , **argv, required=False, type=str2bool, help="whether the TACF has been computed from time derivatives or dipoles (default: %(default)s)", default=True)
    parser.add_argument("-a"  , "--axis"        , **argv, required=False, type=int     , help="axis along compute the FFT (default: %(default)s)", default=1)
    parser.add_argument("-pad", "--padding"     , **argv, required=False, type=int     , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-p"  , "--plot"        , **argv, required=False, type=str     , help="output file for the plot (default: %(default)s)", default='dielectric.pdf')
    # parser.add_argument("-mf" , "--max_freq"    , **argv, required=False, type=float   , help="max frequency in IR plot [THz] (default: %(default)s)", default=NOne)
    parser.add_argument("-fu" , "--freq_unit"   , **argv, required=False, type=str     , help="unit of the frequency in IR plot and output file (default: %(default)s)", default="THz")
    parser.add_argument("-ms" , "--marker_size" , **argv, required=False, type=float   , help="marker size (default: %(default)s)", default=0)
    parser.add_argument("-xl", "--xlim"       , **argv, required=False, type=flist, help="x limits in frequency (default: %(default)s)", default=[None,None])
    parser.add_argument("-yl", "--ylim"       , **argv, required=False, type=flist, help="y limits in frequency (default: %(default)s)", default=[None,None])
    parser.add_argument("-xs", "--x_scale"       , **argv, required=False, type=str, help="x scale (default: %(default)s)", default="log", choices=['linear','log','symlog','logit'])
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the autocorrelation from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    autocorr:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data().T
    print("done")
    print("\tautocorr shape: ",autocorr.shape)
    
    if args.normalize:
        print("\tNormalizing the autocorrelation ... ", end="")
        autocorr = autocorr / autocorr[:,0][:,np.newaxis]
        print("done")

    #------------------#
    if autocorr.ndim == 1:
        print("\tReshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)  
        print("done")
        print("\tdata shape: ",autocorr.shape)

    #------------------#
    if args.input_dipole is not None:
        print("\n\tReading the dipoles from file '{:s}' ... ".format(args.input_dipole), end="")
        args.input_dipole = str(args.input_dipole)
        dipoles:np.ndarray = PhysicalTensor.from_file(file=args.input_dipole).to_data()
        print("done")
        print("\tdipoles shape: ",dipoles.shape)

        #------------------#
        print("\n\tComputing the dipole fluctuations ... ",end="")
        fluctuation:np.ndarray = np.mean(dipoles**2,axis=args.axis)# .sum(axis=-1)
        print("done")
        print("\tfluctuation shape: ",fluctuation.shape)
        # print("\tfluctuation value: {:.2e} +- {:.2e}".format(fluctuation.mean(),fluctuation.std()))
    else:
        print("\n\tTACF is supposed to be normalized and no dipole fluctuations will be computed.")
        # assert np.allclose(autocorr[0,:],3), "TACF is supposed to be normalized."
        fluctuation = None
        
    if args.input_norm is not None:
        print("\n\tReading the normalization constants from file '{:s}' ... ".format(args.input_norm), end="")
        norm = np.loadtxt(args.input_norm)
        print("done")
    else:
        norm = None
        
    if norm is not None and fluctuation is not None:
        assert np.allclose(fluctuation,norm), "Normalization constants are not consistent with the dipole fluctuations."
        
    if fluctuation is None and norm is not None:
        fluctuation = norm

    #------------------#
    print("\n\tComputing the spectrum ... ", end="")
    spectrum, freq = compute_spectrum(autocorr,axis=args.axis,pad=args.padding,method="rfft",dt=args.time_step)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)
    # spectrum = spectrum.real

    #------------------#
    print("\n\tComputing the whole spectrum ... ", end="")
    # spectrum = fluctuation[:,np.newaxis] + 1.j 2*np.pi * freq * spectrum
    omega = 2*np.pi*freq#/args.time_step
    # phases = np.exp(1.j*2*np.pi*freq)-1
    
    if args.remove_zero:
        if fluctuation is not None:
            raise ValueError("Do not provide fluctuation when removing zero frequency.")
            return -1
        if args.derivative:
            with np.errstate(divide='ignore', invalid='ignore'):
                spectrum = 1.j * np.divide(spectrum , omega)
                # spectrum -= np.mean(spectrum[:,1].real)# [:,np.newaxis]
        else:
            spectrum = 1.j* omega*spectrum * fluctuation
            # spectrum -= np.mean(spectrum[:,0].real)# [:,np.newaxis]
    else:
    
        if args.derivative:
            # spectrum = fluctuation[:,np.newaxis] + 1.j* omega * spectrum / (phases*np.conjugate(phases))
            with np.errstate(divide='ignore', invalid='ignore'):
                spectrum = 1 +1.j * np.divide(spectrum , omega)
            if fluctuation is not None:
                fluctuation = np.sum(fluctuation,axis=-1)[:,np.newaxis]
                spectrum *= fluctuation
            # else:
            #     fluctuation = fluctuation[:,np.newaxis]
            
            #     spectrum = fluctuation + tmp
            #     # plt.plot(np.mean(spectrum,axis=0).real,color="red")
            #     # plt.plot(np.mean(spectrum,axis=0).imag,color="blue")
            #     # plt.show()
            # # spectrum = 1 + 1.j * spectrum / omega
        else:
            # spectrum = fluctuation[:,np.newaxis] + 1.j* omega * spectrum
            spectrum = (1 + 1.j* omega*spectrum) * fluctuation
            
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

   #------------------#
    print("\n\tComputing the average over the trajectories ... ", end="")
    N = spectrum.shape[0]
    std:np.ndarray = np.std(spectrum.real,axis=0) + 1.j*np.std(spectrum.imag,axis=0)
    spectrum:np.ndarray = spectrum.mean(axis=0)
    err:np.ndarray = std/np.sqrt(N)
    print("done")
    print("\tspectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"

    # #------------------#
    # print("\n\tComputing the frequencies ... ", end="")
    # # This could require some fixing
    # # Convert timestep to seconds
    # dt = convert(1, "time","femtosecond", "second")
    # # Compute the sampling rate in Hz
    # sampling_rate = 1 / dt
    # # Convert sampling rate to the desired units
    # sampling_rate = convert(sampling_rate, "frequency", "hz", args.freq_unit)
    # # Compute the frequency array
    # freq *= sampling_rate #np.linspace(0, sampling_rate, len(spectrum))
    # print("done")

    #------------------#
    print("\n\tComputing the frequencies ... ", end="")
    freq = get_freq(dt=args.time_step, N=len(spectrum),output_units="THz")
    freq = convert(freq,'frequency','thz',args.freq_unit)
    print("done")

    # assert np.allclose(freq_old,freq), "the frequencies are not the same"

    #------------------#
    print("\n\tSaving the spectrum and the frequecies to file '{:s}' ... ".format(args.output), end="")
    tmp =  np.vstack((freq,spectrum,std,err)).T
    assert tmp.ndim == 2, "this array should have 2 dimensions"
    # assert tmp.shape[1] == 3, "this array should have 3 columns"
    tmp = PhysicalTensor(tmp)
    if str(args.output).endswith("txt"):
        header = \
            f"Col 1: frequency in {args.freq_unit}\n" +\
            f"Col 2: spectrum\n" +\
            f"Col 3: std (over trajectories) of the spectrum\n"+\
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
        mask = ~np.isnan(spectrum.real) & ~np.isnan(spectrum.real)

        if args.xlim is not None:
            xlim_mask = (freq >= args.xlim[0]) & (freq <= args.xlim[1])
        else:
            xlim_mask = np.ones_like(freq,dtype=bool)
        valid_mask = ~np.isnan(spectrum.real) & ~np.isnan(spectrum.imag)
        mask = xlim_mask & valid_mask

        ax.plot(freq[mask],spectrum[mask].real, label="$\\rm \\mathcal{Re} \\chi\\left(\\omega\\right)$",color="blue") #, marker='.', markerfacecolor='blue', markersize=args.marker_size)
        ax.plot(freq[mask],spectrum[mask].imag, label="$\\rm \\mathcal{Im} \\chi\\left(\\omega\\right)$",color="red") #, marker='.', markerfacecolor='red', markersize=args.marker_size)
        # hzero(ax,np.mean(fluctuation),label="fluc.",color="black")

        ylow,yhigh = spectrum - err, spectrum + err
        valid_mask = ~np.isnan(ylow) & ~np.isnan(yhigh)
        mask = xlim_mask & valid_mask
        freq_filtered = freq[mask]
        ylow_filtered = ylow[mask]
        yhigh_filtered = yhigh[mask]
        ax.fill_between(freq_filtered,ylow_filtered.real,yhigh_filtered.real, color='gray', alpha=0.8)#, label='$\\rm \\mathcal{Re}\\left[ \\epsilon \\left(\\omega\\right)\\pm\\sigma\\left(\\omega\\right)\\right]$')
        ax.fill_between(freq_filtered,ylow_filtered.imag,yhigh_filtered.imag, color='gray', alpha=0.8)#, label='$\\rm \\mathcal{Im}\\left[ \\epsilon \\left(\\omega\\right)\\pm\\sigma\\left(\\omega\\right)\\right]$')
        # ylow,yhigh = spectrum - 2*std, spectrum + 2*std
        # ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.5, label='$\\pm2\\sigma$')
        ax.legend(facecolor='white', framealpha=1,edgecolor="black") #loc="upper left"
        ax.set_xlim(args.xlim[0],args.xlim[1])
        # ax.relim()
        # ax.autoscale_view()
        ax.set_ylim(args.ylim[0],args.ylim[1])
        # ax.set_yticks(np.arange(0,1.001,0.2))
        ax.set_xscale(args.x_scale)
        # ax.relim()            # Recompute the data limits for the new xlim
        # ax.autoscale_view()   # Update the view to fit the new data limits
        # ax.set_yscale("log")
        if str(args.freq_unit).lower() == "thz": 
            label = "frequency [THz]"
        elif  str(args.freq_unit).lower() == "inversecm":
            label = r"frequency [cm$^{-1}$]"
        else:
             label = r"frequency [unknown]"
        ax.set_xlabel(label)
        ax.set_ylabel("electric susceptibility [arb. units]")
        ax.grid()
        plt.tight_layout()
        # plt.show()
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