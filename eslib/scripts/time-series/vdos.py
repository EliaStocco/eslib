#!/usr/bin/env python
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.trajectory import AtomicStructures
from eslib.classes.tcf import TimeAutoCorrelation, compute_spectrum, get_freq
from eslib.formatting import esfmt, eslog
from eslib.tools import convert
from eslib.tools import get_files, take

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Vibrational Density Of States (VDOS) from a Molecular Dynamics trajectory."
documentation = \
"The script takes the positions as input and computes the velocities usng `np.gradient`.\n\
Be sure that the trajectory has not been folded in the primitive unit cell.\n\
If that is the case, you can use `unfold.py`."

AXIS_SAMPLES = 0
AXIS_TIME = 1
AXIS_DOFS = 2

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"   , "--input"       , **argv, required=True , type=str  , help="file with the atomic structure")
    parser.add_argument("-if"  , "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-u"   , "--unit"        , **argv, required=False, type=str  , help="unit of the atomic structure (default: %(default)s)" , default="angstrom")
    parser.add_argument("-dt"  , "--time_step"   , **argv, required=True , type=float, help="time step [fs]")
    parser.add_argument("-pad" , "--padding"     , **argv, required=False, type=int  , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    parser.add_argument("-w"   , "--window"      , **argv, required=False, type=str  , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt"  , "--window_t"    , **argv, required=False, type=int  , help="time span of the window [fs] (default: %(default)s)", default=1000)
    parser.add_argument("-o"   , "--output"      , **argv, required=False, type=str  , help="txt/npy output file (default: %(default)s)", default='vdos.txt')
    parser.add_argument("-p"   , "--plot"        , **argv, required=False, type=str  , help="plot file (default: %(default)s)", default=None)
    parser.add_argument("-fu"  , "--freq_unit"   , **argv, required=False, type=str  , help="unit of the frequencies (default: %(default)s)", default="inversecm")
    parser.add_argument("-fmax", "--freq_max"    , **argv, required=False, type=float, help="maximum frequency (default: %(default)s)", default=None)
    
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):
    
    #------------------#
    files = get_files(args.input)
    n_files = len(files)
    print("\t Find {:d} files using '{:s}'".format(n_files,args.input))
    for n,file in enumerate(files):
        print("\t\t {:<3d}/{:>3d}: {:s}".format(n,n_files,file))
    
    #------------------#
    trajectories:List[AtomicStructures] = [None]*n_files
    # with eslog(f"\nReading atomic structures from file '{args.input}'"):
    print()
    for n,file in enumerate(files):
        with eslog(f"Reading atomic structures from file '{file}'"):
            trajectories[n] = AtomicStructures.from_file(file=file)#.get("positions")
            
    #------------------#
    positions = [None]*n_files
    with eslog(f"\nExtracting positions from trajectories "):
        for n,trajectory in enumerate(trajectories):
            positions[n] = trajectory.get("positions")
    positions = np.array(positions)
    print("\t positions shape: ",positions.shape)
    del trajectories # no longer needed
    
    #------------------#
    if args.unit not in ["au","a.u.","atomic_unit","bohr"]:
        with eslog(f"\nConverting positions from '{args.unit}' to 'atomic_unit'"):
            positions = convert(positions,"length",_from=args.unit,_to="atomic_unit")
            
    with eslog(f"\nConverting time step from 'femtosecond' to 'atomic_unit'"):
        args.time_step = convert(args.time_step,"time",_from='femtosecond',_to="atomic_unit")
    
    #------------------#
    with eslog("\nReshaping the positions"):
        positions = positions.reshape((positions.shape[0],positions.shape[1],-1))
    print("\t reshaped positions: ",positions.shape)

    #------------------#
    with eslog("\nComputing the velocities"):
        # the division by the time step is not necessary here
        # because the autocorrelation function will be normalized
        positions = np.gradient(positions,axis=AXIS_TIME)/args.time_step
    print("\t velocities shape: ",positions.shape)    
    
    #------------------#
    with eslog("\nRemoving the mean"):
        positions -= np.mean(positions, axis=AXIS_TIME,keepdims=True)
    print("\t velocities shape: ",positions.shape)    
    
    #------------------#
    with eslog("\nComputing the autocorrelation function"):
        obj = TimeAutoCorrelation(positions)
        autocorr = obj.tcf(axis=AXIS_TIME)
    print("\t autocorr shape: ",autocorr.shape)
    
    fluctuations = np.sum(np.mean(positions**2,axis=AXIS_TIME,keepdims=True),axis=AXIS_DOFS).flatten()
    autocorr_zero = np.take(np.sum(autocorr,axis=AXIS_DOFS,keepdims=True),axis=AXIS_TIME,indices=0).flatten()
    
    assert np.allclose(fluctuations,autocorr_zero), "Something is wrong with the autocorrelation function."
    
    #------------------#
    with eslog("\nSum over all degrees of freedom"):
        autocorr = np.sum(autocorr,axis=AXIS_DOFS)
    print("\t autocorr shape: ",autocorr.shape)
        
    with eslog("\nNormalizing the autocorrelation function"):
        autocorr /= take(autocorr,axis=AXIS_TIME,indices=0,keepdims=True)
    print("\t autocorr shape: ",autocorr.shape)
    assert np.allclose(take(autocorr,axis=AXIS_TIME,indices=0),1), "Something went wrong when normalizing the autocorrelation function."
    
    #------------------#
    if args.window != "none" :
        raw_autocorr = np.copy(autocorr)  
        with eslog(f"\nApplying the '{args.window}' window"):
            func = getattr(np, args.window)
            window = np.zeros(raw_autocorr.shape[args.axis_time])
            M = int(args.window_t / args.time_step)
            window[:M] = func(2*M)[M:]
            window /= window[0] # small correction so that the first value is 1
        
        print("\t window shape: ",window.shape)
        autocorr = raw_autocorr * window
        print("\t autocorr shape: ",autocorr.shape)

    #------------------#
    with eslog("\nComputing the spectrum"):
        spectrum, freq = compute_spectrum(autocorr,axis=AXIS_TIME,pad=args.padding,method="rfft",dt=args.time_step)
    print("\t spectrum shape: :",spectrum.shape, " | the shape can vary depending on the chosen padding")
    
    #------------------#
    with eslog("\nComputing the frequencies"):
        freq = get_freq(dt=args.time_step, N=len(spectrum),output_units="THz")
        freq = convert(freq,'frequency','thz',args.freq_unit)
    print("\t max freq: ",freq[-1],f"{args.freq_unit}")
    
    #------------------#
    with eslog("\nComputing the average over the trajectories"):
        N = spectrum.shape[AXIS_SAMPLES]
        std:np.ndarray      = np.std (spectrum,axis=AXIS_SAMPLES)
        spectrum:np.ndarray = np.mean(spectrum,axis=AXIS_SAMPLES)
        err:np.ndarray      = std/np.sqrt(N-1)
    print("\t spectrum shape: :",spectrum.shape)

    assert spectrum.ndim == 1, "the spectrum does not have 1 dimension"

    #------------------#
    with eslog(f"\nSaving the spectrum and the frequecies to file '{args.output}'"):
        tmp =  np.vstack((freq,spectrum,std,err)).T
        assert tmp.ndim == 2, "this array should have 2 dimensions"
        tmp = PhysicalTensor(tmp)
        if str(args.output).endswith("txt"):
            header = \
                f"Col 1: frequency in {args.freq_unit}\n" +\
                f"Col 2: Vibrational Density Of States (VDOS)\n" +\
                f"Col 3: std (over trajectories) of the VDOS\n"+\
                f"Col 4: error of the VDOS (std/sqrt(N-1), with N = n. of trajectories)"
            tmp.to_file(file=args.output,header=header)
        else:
            tmp.to_file(file=args.output)
    
    #------------------#
    if args.plot is not None:
        with eslog(f"\nCreating plot and saving it to '{args.plot}'"):
            
            if args.freq_max is not None:
                ii = np.where(freq <= args.freq_max)[0]
                freq = freq[ii]
                spectrum = spectrum[ii]
                err = err[ii]
            
            fig, ax = plt.subplots(1, figsize=(6, 4))
            ax.plot(freq,spectrum,color="blue", marker='.', markerfacecolor='blue')
            ylow,yhigh = spectrum - err, spectrum + err
            ax.fill_between(freq,ylow,yhigh, color='gray', alpha=0.8)

            ax.legend(loc="upper left",facecolor='white', framealpha=1,edgecolor="black")
            ax.set_xlim(0,args.freq_max)
            # ax.set_ylim(args.ylim[0],args.ylim[1])
            # if args.normalize:
            #     ax.set_yticks(np.arange(0,1.001,0.2))
            ax.set_ylabel("VDOS [arb. units]")
            ax.set_xlabel(f"frequency [{args.freq_unit}]")        
            ax.grid()
            plt.tight_layout()
            plt.savefig(args.plot,bbox_inches="tight")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()