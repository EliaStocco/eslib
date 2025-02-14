#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd
from ase import Atoms
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from eslib.functions import extract_number_from_filename
from eslib.input import flist, slist, itype
from eslib.formatting import esfmt
from eslib.classes.physical_tensor import load_data
from eslib.io_tools import save2json
from eslib.classes.atomic_structures import AtomicStructures
from eslib.tools import convert
from eslib.plot import legend_options
from eslib.physics import debye_real, Debye_unc, Debye_fit
from eslib.classes.aseio import integer_to_slice_string
from eslib.mathematics import mean_std_err

#---------------------------------------#
description = "Fit the Debye model for susceptibility with higher harmonics."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file with the atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-N" , "--N_harmonics" , **argv, required=False, type=int  , help="maximum harmonics (default: %(default)s)", default=1)
    parser.add_argument("-d" , "--dipole"      , **argv, required=True , type=slist, help="dipole files [eang]")
    parser.add_argument("-t" , "--time"        , **argv, required=True , type=slist, help="time files [ps]")
    parser.add_argument("-w" , "--frequencies" , **argv, required=True , type=flist, help="list of frequencies [GHz]")
    parser.add_argument("-E" , "--Efields"     , **argv, required=True , type=flist, help="list of electric field intensities [V/ang]")
    parser.add_argument("-f" , "--folder"      , **argv, required=False, type=str  , help="output folder (default: %(default)s)", default="tmp")
    parser.add_argument("-fo", "--fit_output"  , **argv, required=False, type=str  , help="output file with the fit results (default: %(default)s)", default="fit.json")
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output file (default: %(default)s)", default="debye.csv")
    parser.add_argument("-p" , "--plot"        , **argv, required=False, type=str  , help="plot (default: %(default)s)", default="debye.pdf")
    return parser 

#---------------------------------------#
def get_data(args):
    
    #------------------#
    data = {k:None for k in args.frequencies}
    index = integer_to_slice_string(args.index)
    
    #------------------#
    for n,key in enumerate(args.frequencies):
        print(f"\n\tReading files for frequency {key}GHz:")
        
        
        time_files = glob.glob(args.time[n])
        dipole_files = glob.glob(args.dipole[n])
        
        if type(time_files) == str:
            time_files = [time_files]
        elif len(time_files) == 1:
            pass
        else:
            time_files = [ time_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in time_files ])) ]
        if type(dipole_files) == str:
            dipole_files = [dipole_files]
        elif len(dipole_files) == 1:
            pass
        else:        
            dipole_files = [ dipole_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in dipole_files ])) ]
        
        N = len(time_files)
        data[key] = {
            "time" : [None]*N,
            "dipole" : [None]*N
        }
        
        for i,(tf,df) in enumerate(zip(time_files,dipole_files)):
            
            print(f"\t - reading file '{tf}'")
            time = load_data(tf)/1000 # ps --> ns
            assert time.ndim == 1, f"File {tf} is not a 1D array"
            
            print(f"\t - reading file '{df}'")
            dipole = load_data(df) 
            assert dipole.ndim == 2, f"File {df} is not a 2D array"
            assert dipole.shape[1] == 3, f"File {df} is not a 3D array"
            
            assert time.shape[0] == dipole.shape[0], f"Time and dipole have different shapes: {time.shape} != {dipole.shape}"
            
            time = time[index]
            dipole = dipole[index,:]
            
            data[key]["time"][i] = time
            data[key]["dipole"][i] = dipole
            
    return data

#---------------------------------------#
def high_harmonics(omega,N_harmonics,time,*argv):
    out = np.zeros_like(time)
    for n in range(N_harmonics):
        # print("harmonic:",2*n+1)
        phi = 2*np.pi*omega*time*(2*n+1) # only odd harmonics
        ReP, ImP = argv[n*2], argv[n*2+1]
        out += ReP*np.cos(phi)-ImP*np.sin(phi)
    return out

#---------------------------------------#
def divide_array(arr, n):
    # Calculate the indices to split the array
    indices = np.linspace(0, len(arr), n + 1, dtype=int)
    # Use the indices to split the array
    chunks = [arr[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
    
    # Assert that appending the chunks reconstructs the original array
    reconstructed = np.concatenate(chunks)
    assert np.array_equal(reconstructed, arr), "Reconstructed array does not match the original"
    
    return chunks

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    assert args.N_harmonics > 0, "Maximum harmonics must be positive"
    
    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    volume = atoms.get_volume()
    print("\tvolume: {:f} A^3".format(volume))
    
    assert len(args.frequencies) == len(args.Efields), "Frequencies and Efields have different sizes: {:d} != {:d}".format(len(args.frequencies),len(args.Efields))
    assert len(args.frequencies) == len(args.time), "Frequencies and time have different sizes: {:d} != {:d}".format(len(args.frequencies),len(args.time))
    assert len(args.time) == len(args.dipole), "Time and dipole have different sizes: {:d} != {:d}".format(len(args.time),len(args.dipole))
    
    #------------------#
    data = get_data(args)
    
    #------------------#
    fit = {}
    print()
    for n,freq in enumerate(args.frequencies):
        
        print(f"\tFit for frequency frequency {freq}GHz ... ",end="")
        
        fit[freq] = {}
        
        funcW = lambda *argv: high_harmonics(freq,args.N_harmonics,*argv)
        try:
            time = np.append(*data[freq]["time"]).flatten() # ns
        except:
            time = np.asarray(data[freq]["time"]).flatten() # ns
        try:
            pol = np.append(*data[freq]["dipole"]).reshape((-1,3))[:,2].flatten() # only z-component
        except:
            pol = np.asarray(data[freq]["dipole"]).reshape((-1,3))[:,2].flatten() # only z-component
        p0 = [1,1]*args.N_harmonics
        
        N = int(freq)
        times = divide_array(time,N)
        pols = divide_array(pol,N)
        
        fit[freq]["raw"] = {}
        fit[freq]["raw"]["popt"] = [None]*len(times)
        fit[freq]["raw"]["pcov"] = [None]*len(times)
        fit[freq]["raw"]["perr"] = [None]*len(times)
        
        for k,(T,P) in enumerate(zip(times,pols)):
            assert len(T) == len(P), f"Time and pol have different sizes: {len(T)} != {len(P)}"
            popt, pcov = curve_fit(funcW,T,P,p0=p0) # bounds=((0,0), (+np.inf,+np.inf))
            fit[freq]["raw"]["popt"][k] = popt
            fit[freq]["raw"]["pcov"][k] = pcov
            fit[freq]["raw"]["perr"][k] = np.sqrt(np.diag(pcov))
        
        popts = np.asarray([ popt for popt in fit[freq]["raw"]["popt"] ])
        mean, std, err = mean_std_err(popts,axis=0)
        fit[freq]["popt"] = mean
        fit[freq]["pstd"] = std
        fit[freq]["perr"] = err
        
        # popt, pcov = curve_fit(funcW,time,pol,p0=p0) # bounds=((0,0), (+np.inf,+np.inf))
        # fit[freq]["popt"] = popt
        # fit[freq]["pcov"] = pcov
        # fit[freq]["perr"] = np.sqrt(np.diag(pcov))
        
        print("done")
        
        if args.folder is not None:
            os.makedirs(args.folder, exist_ok=True)
            ofile = f"{args.folder}/W={freq}GHz.pdf"
            print(f"\tSaving fit plot to file {ofile} ... ",end="")
            
            Nsamples = len(data[freq]["time"])
            fig,axes = plt.subplots(Nsamples,figsize=(5,3*Nsamples),sharex=True)
            if Nsamples == 1:
                axes = [axes]
                
            for i in range(Nsamples):
                ax = axes[i]
                time = data[freq]["time"][i] # ns
                pol = data[freq]["dipole"][i].reshape((-1,3))[:,2] # only z-component
                ax.plot(time,pol,color="blue",label="data")
                yfit = funcW(time,*popt)
                ax.plot(time,yfit,color="red",label="fit")
                # ax.plot(time,pol-yfit,color="green",label="data-fit")
                ax.grid()
                ax.legend(loc="upper right")
            fig.supxlabel("Time [ns]")
            fig.supylabel("Dipole [eang]")
            plt.tight_layout()        
            plt.savefig(ofile,dpi=300,bbox_inches="tight")
            plt.close()
            
            print("done\n")
        
    #------------------#
    # eang
    err = np.asarray([ a["perr"][0] for a in fit.values() ])
    np.nan_to_num(err, copy=False, nan=1e-8)
    ReDipole = unp.uarray(
        nominal_values=np.asarray([ a["popt"][0] for a in fit.values() ]),
        std_devs=err
    )
    
    err = np.asarray([ a["perr"][1] for a in fit.values() ])
    np.nan_to_num(err, copy=False, nan=1e-8)
    ImDipole = unp.uarray(
        nominal_values=np.asarray([ a["popt"][1] for a in fit.values() ]),
        std_devs=err
    )
    # dipole = ReDipole+1j*ImDipole
    ReDipole = convert(ReDipole,"electric-dipole","eang","atomic_unit")
    ImDipole = convert(ImDipole,"electric-dipole","eang","atomic_unit")
    volume = convert(volume,"volume","angstrom3","atomic_unit")
    RePol = ReDipole/volume
    ImPol = ImDipole/volume
    
    # phase = unp.uarray(
    #     nominal_values=np.asarray([ a["popt"][1] for a in fit.values() ]),
    #     std_devs=np.asarray([ a["perr"][1] for a in fit.values() ])
    # )
    
    Efields = np.asarray(args.Efields)
    Efields = convert(Efields,"electric-field","v/ang","atomic_unit")
    
    # epsilon_0 = 8.8541878128  # e-12 C^2/Nm^2
    # C2 = convert(1,"charge","coulomb","atomic_unit")**2
    # m2 = convert(1,"length","meter","atomic_unit")**2
    # N = convert(1,"force","newton","atomic_unit")
    # factor = C2/(N*m2)
    # epsilon_0 *= factor
    # epsilon_0 *= 1e-12
    
    epsilon_0 = 1./(4*np.pi)
    
    print("\tVacuum permittivity (E) in Hartree atomic units: ",epsilon_0)
    print("\t4piE: ",4*np.pi*epsilon_0)
    
    # Xreal = np.abs(polarization*unp.cos(phase)/(Efields*epsilon_0))
    # Ximag = np.abs(polarization*unp.sin(phase)/(Efields*epsilon_0))
    
    Xreal = np.abs(RePol/(Efields*epsilon_0))
    Ximag = np.abs(ImPol/(Efields*epsilon_0))
    
    df = pd.DataFrame(columns=["freq [GHz]","Xreal","Ximag","Xreal-err","Ximag-err"])
    args.frequencies = np.asarray(args.frequencies)
    df["freq [GHz]"] = args.frequencies
    df["Xreal"] = unp.nominal_values(Xreal)
    df["Ximag"] = unp.nominal_values(Ximag)
    df["Xreal-err"] = unp.std_devs(Xreal)
    df["Ximag-err"] = unp.std_devs(Ximag)
    df.to_csv(args.output,index=False)
    
    #------------------#
    popt, pcov = curve_fit(Debye_fit,
        xdata=np.asarray([args.frequencies,args.frequencies]).flatten(),
        ydata=np.asarray([unp.nominal_values(Xreal),unp.nominal_values(Ximag)]).T.flatten(),
        sigma=np.asarray([unp.std_devs(Xreal),unp.std_devs(Ximag)]).T.flatten(),
        bounds=((0,0),(np.inf, np.inf))
    )
    fit["debye"] = {}
    fit["debye"]["popt"] = popt
    fit["debye"]["pcov"] = pcov
    fit["debye"]["perr"] = np.sqrt(np.diag(pcov))
    
    parameters = unp.uarray(
        nominal_values=np.asarray(fit["debye"]["popt"]),
        std_devs=np.asarray(fit["debye"]["perr"])
    )
    
    epsilon_r = debye_real(0,parameters[0],parameters[1])
    fit["epsilon_r"] = {
        "value" : unp.nominal_values(epsilon_r),
        "err"   : unp.std_devs(epsilon_r)
    }
    
    #------------------#
    print(f"\n\tSaving fit results to file '{args.fit_output}' ... ", end="")
    save2json(args.fit_output,fit)
    print("done")
    
    #------------------#
    if args.plot is not None:
        fig,ax = plt.subplots(figsize=(4,3))
        freq = np.asarray(args.frequencies)
        ax.errorbar(args.frequencies,unp.nominal_values(Xreal),yerr=unp.std_devs(Xreal),fmt=".",color="red",label=r"Re$\chi$ (data)")
        ax.errorbar(args.frequencies,unp.nominal_values(Ximag),yerr=unp.std_devs(Ximag),fmt=".",color="blue",label=r"Im$\chi$ (data)")
        
        freq = np.linspace(0.1,1000,10000)
        yreal,yimag = Debye_unc(freq,parameters[0],parameters[1])
        ax.plot(freq,unp.nominal_values(yreal),color="red",alpha=1,linewidth=0.1,label=r"Re$\chi$ (fit)")
        ax.plot(freq,unp.nominal_values(yimag),color="blue",alpha=1,linewidth=0.1,label=r"Im$\chi$ (fit)")
        
        ax.fill_between(freq,
            unp.nominal_values(yreal)-unp.std_devs(yreal),
            unp.nominal_values(yreal)+unp.std_devs(yreal),
            color="red",alpha=0.5,linewidth=0
        )
        ax.fill_between(freq,
            unp.nominal_values(yimag)-unp.std_devs(yimag),
            unp.nominal_values(yimag)+unp.std_devs(yimag),
            color="blue",alpha=0.5,linewidth=0
        )
        
        ax.grid()
        ax.legend(loc="center left",**legend_options)
        ax.set_ylabel(r"susceptibility $\chi\left(\omega\right)$")
        ax.set_xlabel(r"frequency $\nu$ [GHz]")
        ax.set_xlim(0.1,1000)
        ax.set_xscale("log")
        plt.tight_layout()        
        # plt.show()
        plt.savefig(args.plot) 
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
    
