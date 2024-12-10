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
from eslib.input import flist, slist
from eslib.formatting import esfmt
from eslib.classes.physical_tensor import load_data
from eslib.io_tools import save2json
from eslib.classes.atomic_structures import AtomicStructures
from eslib.tools import convert
from eslib.plot import legend_options

#---------------------------------------#
description = "Fit the Debye model for susceptibility."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file with the atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
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
    
    #------------------#
    for n,key in enumerate(args.frequencies):
        print(f"\n\tReading files for frequency {key}GHz:")
        
        
        time_files = glob.glob(args.time[n])
        dipole_files = glob.glob(args.dipole[n])
        
        time_files = [ time_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in time_files ])) ]
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
            
            data[key]["time"][i] = time
            data[key]["dipole"][i] = dipole
            
    return data

#---------------------------------------#
def function(omega,time,P,phi):
    out = P*np.cos(2*np.pi*omega*time-phi)# +ImP*np.cos(omega*time)
    return out

#---------------------------------------#
def Debye(w:np.ndarray,chi0:float,tau:float)->np.ndarray:
    return chi0/(1-1j*w*tau)
    
def Debye_fit(_w,chi0,tau):
    w = np.zeros_like(_w).reshape((-1,2))
    w[:,0] = _w[:int(len(_w)/2)]
    w[:,1] = _w[int(len(_w)/2):]
    assert np.allclose(w[:,0],w[:,1]), "w[:,0] != w[:,1]"
    w = w[:,0]
    res = Debye(w,chi0,tau)
    out = np.zeros((len(w),2),dtype=float)
    out[:,0] = np.real(res)
    out[:,1] = np.imag(res)
    return out.flatten()

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
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
    for n,omega in enumerate(args.frequencies):
        
        print(f"\tFit for frequency frequency {omega}GHz ... ",end="")
        
        fit[omega] = {}
        
        funcW = lambda time,ReP,ImP: function(omega,time,ReP,ImP)
        time = np.append(*data[omega]["time"]) # ns
        pol = np.append(*data[omega]["dipole"]).reshape((-1,3))[:,2] # only z-component
        popt, pcov = curve_fit(funcW,time,pol,bounds=((0,0), (+np.inf,2*np.pi)))
        fit[omega]["popt"] = popt
        fit[omega]["pcov"] = pcov
        fit[omega]["perr"] = np.sqrt(np.diag(pcov))
        
        print("done")
        
        if args.folder is not None:
            os.makedirs(args.folder, exist_ok=True)
            ofile = f"{args.folder}/W={omega}GHz.pdf"
            print(f"\tSaving fit plot to file {ofile} ... ",end="")
            
            Nsamples = len(data[omega]["time"])
            fig,axes = plt.subplots(Nsamples,figsize=(5,3*Nsamples),sharex=True)
            if Nsamples == 1:
                axes = [axes]
                
            for i in range(Nsamples):
                ax = axes[i]
                time = data[omega]["time"][i] # ns
                pol = data[omega]["dipole"][i].reshape((-1,3))[:,2] # only z-component
                ax.plot(time,pol,color="blue",label="data")
                ax.plot(time,funcW(time,*popt),color="red",label="fit")
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
    dipole = unp.uarray(
        nominal_values=np.asarray([ a["popt"][0] for a in fit.values() ]),
        std_devs=np.asarray([ a["perr"][0] for a in fit.values() ])
    )
    dipole = convert(dipole,"electric-dipole","eang","atomic_unit")
    volume = convert(volume,"volume","angstrom3","atomic_unit")
    polarization = dipole/ volume
    
    phase = unp.uarray(
        nominal_values=np.asarray([ a["popt"][1] for a in fit.values() ]),
        std_devs=np.asarray([ a["perr"][1] for a in fit.values() ])
    )
    
    Efields = np.asarray(args.Efields)
    Efields = convert(Efields,"electric-field","v/ang","atomic_unit")
    
    epsilon_0 = 8.8541878128  # e-12 C^2/Nm^2
    C2 = convert(1,"charge","coulomb","atomic_unit")**2
    m2 = convert(1,"length","meter","atomic_unit")**2
    N = convert(1,"force","newton","atomic_unit")
    factor = C2/(N*m2)
    epsilon_0 *= factor
    epsilon_0 *= 1e-12
    
    print("\tVacuum permittivity (E) in Hartree atomic units: ",epsilon_0)
    print("\t4piE: ",4*np.pi*epsilon_0)

    Xreal = polarization*unp.cos(phase)/(Efields*epsilon_0)
    Ximag = polarization*unp.sin(phase)/(Efields*epsilon_0)
    
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
    
    #------------------#
    print(f"\n\tSaving fit results to file '{args.fit_output}' ... ", end="")
    save2json(args.fit_output,fit)
    print("done")
    
    #------------------#
    if args.plot is not None:
        fig,ax = plt.subplots(figsize=(4,3))
        omega = np.asarray(args.frequencies)
        ax.errorbar(args.frequencies,unp.nominal_values(Xreal),yerr=unp.std_devs(Xreal),fmt=".",color="red",label=r"Re$\chi$ (data)")
        ax.errorbar(args.frequencies,unp.nominal_values(Ximag),yerr=unp.std_devs(Ximag),fmt=".",color="blue",label=r"Im$\chi$ (data)")
        
        omega = np.linspace(0.1,1000,10000)
        y = Debye(omega,*popt)
        ax.plot(omega,y.real,color="red",alpha=0.5,label=r"Re$\chi$ (fit)")
        ax.plot(omega,y.imag,color="blue",alpha=0.5,label=r"Im$\chi$ (fit)")
        
        ax.grid()
        ax.legend(loc="center left",**legend_options)
        ax.set_ylabel(r"susceptibility $\chi\left(\omega\right)$")
        ax.set_xlabel(r"frequency $\nu$ [GHz]")
        ax.set_xlim(0.1,1000)
        ax.set_xscale("log")
        plt.tight_layout()        
        plt.savefig(args.plot) 
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
    
