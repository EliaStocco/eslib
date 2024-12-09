#!/usr/bin/env python
import glob
import numpy as np
from eslib.functions import extract_number_from_filename
from eslib.input import flist, slist
from eslib.formatting import esfmt
from eslib.classes.physical_tensor import load_data
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#---------------------------------------#
description = "Fit the Debye model for susceptibility."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-d" , "--dipole"      , **argv, required=True , type=slist, help="dipole files [eang]")
    parser.add_argument("-t" , "--time"        , **argv, required=True , type=slist, help="time files [ps]")
    parser.add_argument("-w" , "--frequencies" , **argv, required=True , type=flist, help="list of frequencies [GHz]")
    parser.add_argument("-E" , "--Efields"     , **argv, required=True , type=flist, help="list of electric field intensities [V/ang]")
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output file (default: %(default)s)", default="debye.csv")
    parser.add_argument("-p" , "--plot"        , **argv, required=False, type=str  , help="plot (default: %(default)s)", default="debye.pdf")
    return parser 

def get_data(args):
    
    #------------------#
    N = len(args.frequencies)
    data = {k:None for k in args.frequencies}
    
    #------------------#
    for n,key in enumerate(args.frequencies):
        print(f"\tFrequency: {key}GHz")
        data[key] = {
            "time" : [None]*N,
            "dipole" : [None]*N
        }
        
        time_files = glob.glob(args.time[n])
        dipole_files = glob.glob(args.dipole[n])
        
        time_files = [ time_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in time_files ])) ]
        dipole_files = [ dipole_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in dipole_files ])) ]
        
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

def function(omega,time,P,phi):
    out = P*np.sin(omega*time+phi)# +ImP*np.cos(omega*time)
    return out

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    assert len(args.frequencies) == len(args.Efields), "Frequencies and Efields have different sizes: {:d} != {:d}".format(len(args.frequencies),len(args.Efields))
    assert len(args.frequencies) == len(args.time), "Frequencies and time have different sizes: {:d} != {:d}".format(len(args.frequencies),len(args.time))
    assert len(args.time) == len(args.dipole), "Time and dipole have different sizes: {:d} != {:d}".format(len(args.time),len(args.dipole))
    
    #------------------#
    data = get_data(args)
    
    #------------------#
    fit = {}
    for n,omega in enumerate(args.frequencies):
        fit[omega] = {}
        
        funcW = lambda time,ReP,ImP: function(omega,time,ReP,ImP)
        time = np.append(*data[omega]["time"])
        pol = np.append(*data[omega]["dipole"]).reshape((-1,3))[:,2] # only z-component
        popt, pcov = curve_fit(funcW,time,pol,bounds=((0,0), (+np.inf,2*np.pi)))
        fit[omega]["popt"] = popt
        fit[omega]["pcov"] = pcov
        pass
    
    modulus = np.asarray([ a["popt"][0] for a in fit.values() ])
    phase =  np.asarray([ a["popt"][1] for a in fit.values() ])
    
    real = modulus*np.cos(phase)
    imag = modulus*np.sin(phase)
    
    #------------------#
    if args.plot is not None:
        fig,ax = plt.subplots(figsize=(3,3))
        omega = np.asarray(args.frequencies)
        ax.plot(omega,real,color="blue",label="real")
        ax.plot(omega,imag,color="red",label="imag")
        ax.grid()
        ax.legend()
        plt.tight_layout()        
        plt.savefig(args.plot) 
    
        

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
    
