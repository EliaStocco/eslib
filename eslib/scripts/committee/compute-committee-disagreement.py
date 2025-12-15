#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt, float_format
from eslib.io_tools import pattern2sorted_files, pattern2data

#---------------------------------------#
description = "Compute the committee disagreement of an array."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input" ,**argv, type=str, required=True , help="FHI-aims output file")
    parser.add_argument("-c", "--components" ,**argv, type=int, required=True , help="number of components of the array")
    parser.add_argument("-o", "--output",**argv, type=str, required=False, help="output file (default: %(default)s)", default="disagreement")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    files = pattern2sorted_files(args.input)
    print(f"\tFound {len(files)} files matching the pattern '{args.input}'")
    print(f"\tProcessing files to extract data ...",end="")
    data = pattern2data(args.input)
    print("done")
    print(f"\tdata.shape: {data.shape}")
    
    #------------------#
    Nsamples = data.shape[0]
    Ncomponents = int(args.components)
    Nstructures = data.shape[1]
    data = data.reshape((Nsamples,Nstructures,-1,Ncomponents))
    Natoms = data.shape[2]
    print()
    print(f"\tdata.shape: {data.shape}")
    print(f"\tnumber of samples   : {Nsamples}")
    print(f"\tnumber of structures: {Nstructures}")
    print(f"\tnumber of atoms     : {Natoms}")
    print(f"\tnumber of components: {Ncomponents}")
    
    #------------------#
    print(f"\tComputing committee disagreement ... ", end="")
    mean = np.mean(data,axis=0)
    data -= mean
    err2 = np.square(data)
    err2 = np.sum(err2,axis=3)  # (Nsamples,Nstructures,Natoms)
    err2 = np.mean(err2,axis=0) # (Nstructures,Natoms)
    err = np.sqrt(err2)         # (Nstructures,Natoms)
    disagreement = np.mean(err,axis=1) # (Nstructures,)
    print("done")
    print(f"\tdisagreement.shape: {disagreement.shape}")
    print()
    print("\tmax disagreement: ",np.max(disagreement))
    print("\tmin disagreement: ",np.min(disagreement))
    print("\tavg disagreement: ",np.mean(disagreement))
    print("\tstd disagreement: ",np.std(disagreement))
    
    #------------------#
    print(f"\tSaving disagreement to file '{args.output}' ... ", end="")
    np.savetxt(args.output, disagreement,fmt=float_format)
    print("done")
    
    return 

#---------------------------------------#
if __name__ == "__main__":
    main()

