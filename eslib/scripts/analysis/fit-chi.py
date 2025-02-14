#!/usr/bin/env python
import glob
import numpy as np
import pandas as pd
from ase import Atoms
from scipy.optimize import curve_fit
from eslib.functions import extract_number_from_filename
from eslib.input import flist, slist, itype
from eslib.formatting import esfmt
from eslib.classes.physical_tensor import load_data
from eslib.classes.atomic_structures import AtomicStructures
from eslib.mathematics import mean_std_err
from eslib.tools import pandas_append
from eslib.tools import convert

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
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output folder (default: %(default)s)", default=".")
    return parser 

#---------------------------------------#
def get_data(args):
    
    df = pd.DataFrame(columns=["index","omega","efield","file-time","file-dipole"])
    
    #------------------#
    for n,key in enumerate(args.frequencies):
        # print(f"\n\tReading files for frequency {key}GHz:")
        
        
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
            
        assert len(time_files) == 1
        assert len(dipole_files) == 1
          
        new_row = {
            "index" : n,
            "omega" : args.frequencies[n],
            "efield": args.Efields[n],
            "file-time": time_files[0],
            "file-dipole": dipole_files[0]
        }
        
        df = pandas_append(df,new_row)
        
    return df
        
    #     for i,(tf,df) in enumerate(zip(time_files,dipole_files)):
            
    #         print(f"\t - reading file '{tf}'")
    #         time = load_data(tf)/1000 # ps --> ns
    #         assert time.ndim == 1, f"File {tf} is not a 1D array"
            
    #         print(f"\t - reading file '{df}'")
    #         dipole = load_data(df) 
    #         assert dipole.ndim == 2, f"File {df} is not a 2D array"
    #         assert dipole.shape[1] == 3, f"File {df} is not a 3D array"
            
    #         assert time.shape[0] == dipole.shape[0], f"Time and dipole have different shapes: {time.shape} != {dipole.shape}"
            
    #         time = time[index]
    #         dipole = dipole[index,:]
            
    #         data[key]["time"][i] = time
    #         data[key]["dipole"][i] = dipole
            
    # return data

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
    df = pd.DataFrame(columns=["index","freq [GHz]","Xreal","Ximag","Xreal-err","Ximag-err"])
    print()
    for n,row in data.iterrows():
        
        freq = row["omega"]
        Efield = row["efield"]
        file_time = row["file-time"]
        file_dipole = row["file-dipole"]
        
        print(f"\t{n}) fit for:")
        print(f"\t  - omega {freq}GHz")
        print(f"\t  - Efield {Efield}V/ang")
        print(f"\t  - time from '{file_time}'")
        print(f"\t  - dipole from '{file_dipole}'")
        print()
        
        time = load_data(file_time)/1000 # ps --> ns
        assert time.ndim == 1, f"File {file_time} is not a 1D array"
        
        dipole = load_data(row["file-dipole"]) 
        assert dipole.ndim == 2, f"File {file_dipole} is not a 2D array"
        assert dipole.shape[1] == 3, f"File {file_dipole} is not a 3D array"
        pol = dipole[:,2]/volume
       
        pol = convert(pol,"polarization","e/ang2","atomic_unit")
        efield = convert(row["efield"],"electric-field","V/ang","atomic_unit") 
        pol /= efield
        epsilon_0 = 1./(4*np.pi)
        pol /= epsilon_0
    
        # fit[freq] = {}
        
        funcW = lambda *argv: high_harmonics(freq,args.N_harmonics,*argv)
        # try:
        #     time = np.append(*data[freq]["time"]).flatten() # ns
        # except:
        #     time = np.asarray(data[freq]["time"]).flatten() # ns
        # try:
        #     pol = np.append(*data[freq]["dipole"]).reshape((-1,3))[:,2].flatten() # only z-component
        # except:
        #     pol = np.asarray(data[freq]["dipole"]).reshape((-1,3))[:,2].flatten() # only z-component
        p0 = [1,1]*args.N_harmonics
        
        N = int(freq)
        times = divide_array(time,N)
        pols = divide_array(pol,N)
        
        tmp = {}
        tmp["popt"] = [None]*len(times)
        tmp["pcov"] = [None]*len(times)
        tmp["perr"] = [None]*len(times)
        
        for k,(T,P) in enumerate(zip(times,pols)):
            assert len(T) == len(P), f"Time and pol have different sizes: {len(T)} != {len(P)}"
            popt, pcov = curve_fit(funcW,T,P,p0=p0) # bounds=((0,0), (+np.inf,+np.inf))
            tmp["popt"][k] = popt
            tmp["pcov"][k] = pcov
            tmp["perr"][k] = np.sqrt(np.diag(pcov))
        
        popts = np.asarray([ popt for popt in tmp["popt"] ])
        mean, std, err = mean_std_err(popts,axis=0)
        mean = np.abs(mean)
        
        new_row = {
            "index"     : n, 
            "freq [GHz]": freq ,
            "Xreal"     : mean[0],
            "Ximag"     : mean[1],
            "Xreal-std" : std[0],
            "Ximag-std" : std[1],
            "Xreal-err" : err[0],
            "Ximag-err" : err[1]
        }
        df = pandas_append(df,new_row)
        
        print(f"\t  - Re X: {mean[0]}")
        print(f"\t  - Im X: {mean[1]}")
        print()
        
    print("done")
    
    #------------------#
    ofile = f"{args.output}/chi.csv"
    print("\n\tSaving fit results to file '{:s}' ... ".format(ofile), end="")
    df.to_csv(ofile,index=False)
    print("done")
    
    #------------------#
    unique_freqs = df["freq [GHz]"].unique()
    results = pd.DataFrame(columns=["freq [GHz]","Xreal","Ximag","Xreal-std","Ximag-std","Xreal-err","Ximag-err"])

    for freq in unique_freqs:
        subset = df[df["freq [GHz]"] == freq]
        
        Xreal = np.asarray(subset["Xreal"])
        Ximag = np.asarray(subset["Ximag"])
        
        Rmean, Rstd, Rerr = mean_std_err(Xreal,axis=0)
        Imean, Istd, Ierr = mean_std_err(Ximag,axis=0)
        
        new_row = {
            "freq [GHz]": freq,
            "Xreal": Rmean,
            "Ximag": Imean,
            "Xreal-std": Rstd,
            "Ximag-std": Istd,
            "Xreal-err": Rerr,
            "Ximag-err": Ierr
        }
        
        results = pandas_append(results,new_row)

    #------------------#
    ofile = f"{args.output}/results.csv"
    print("\n\tSaving averaged (over frequencies) results to file '{:s}' ... ".format(ofile), end="")
    results.to_csv(ofile,index=False)
    print("done")
        

#---------------------------------------#
if __name__ == "__main__":
    main()

