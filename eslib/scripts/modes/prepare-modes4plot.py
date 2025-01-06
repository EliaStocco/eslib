#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from eslib.formatting import esfmt
from eslib.io_tools import pattern2sorted_files
from eslib.mathematics import mean_std_err

#---------------------------------------#
description = "Preprocess the data of the phonon projections and compute the average over the trajectories."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , **argv, required=True, type=str, help="input files")
    parser.add_argument("-o", "--output", **argv, required=True, type=str, help="output folders")
    return parser

@esfmt(prepare_args,description)
def main(args):

    print("\tSearching for files ... ", end="")
    files = pattern2sorted_files(args.input)
    print("done")
    print("\tn of files: ",len(files))
    
    #------------------#
    Nsnapshot = None
    Nmodes = None
    Ntrajectories = len(files)
    
    #------------------#
    # Read the input files
    test = np.loadtxt(files[0],delimiter=",")
    Nsnapshot = test.shape[0]
    Nmodes = test.shape[1]
    data = np.full((Ntrajectories,Nsnapshot,Nmodes),np.nan)
    del test 
    
    print("\t - n. of snapshots: ",Nsnapshot)
    print("\t - n. of modes    : ",Nmodes)
    print("\t - n. of trajectories: ",Ntrajectories)
    
    #------------------#
    print("\n\tReading file:")
    for n,file in enumerate(files):
        print(f"\t - {file} ... ", end="")
        tmp = np.loadtxt(file,delimiter=",")
        assert tmp.shape == (Nsnapshot,Nmodes), "The shape of the data must be ({:d},{:d}) but got ({:d},{:d})".format(Nsnapshot,Nmodes,tmp.shape[0],tmp.shape[1])
        data[n] = tmp
        print("done")
    
    # print("\n\tdata.shape: ",data.shape)
    assert not np.any(np.isnan(data)), "The data contains NaNs"
    
    print("\n\tTransposing the data ... ",end="")
    data = np.moveaxis(data,2,0) # modes to 0
    data = np.moveaxis(data,2,1) # snapshots to 1, and trajectories to 2
    # data = np.transpose(data,(1,2,0))
    print("done")
    print("\tdata.shape: ",data.shape)
   
    #------------------#
    os.makedirs(args.output,exist_ok=True)
    for n in range(Nmodes):
        mode = data[n]
        mean, std, err = mean_std_err(mode,axis=1)
        ofile = os.path.join(args.output,f"mode.n={n}.csv")
        print(f"\t - writing mode {n} to '{ofile}' ... ",end="")
        df = pd.DataFrame(np.column_stack((mean,std,err)),columns=["mean","std","err"])
        df.to_csv(ofile,index=False)
        print("done")

if __name__ == "__main__":
    main()