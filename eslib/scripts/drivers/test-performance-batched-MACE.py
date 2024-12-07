#!/usr/bin/env python
from ase.io import read
import numpy as np
import time
import torch
from eslib.formatting import esfmt
from eslib.classes.models.mace_model import MACEModel
import matplotlib.pyplot as plt 
from eslib.plot import legend_options

#---------------------------------------#
# Description of the script's purpose
description = "Run FileIOBatchedMACE."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="file with the atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="file format of the atomic structure (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--model"        , **argv, required=False, type=str, help="file with the MACE model (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--max_batch"    , **argv, required=False, type=int, help="maximum batch size (default: %(default)s)", default=64)
    parser.add_argument("-d" , "--discard"      , **argv, required=False, type=int, help="first calculations to be discarded (default: %(default)s)", default=4)
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str, help="output file with the plot (default: %(default)s)", default="report.pdf")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tCuda available: ",torch.cuda.is_available())
    
    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=0)
    atoms.calc = None # I don't know if I actually need this line
    print("done",flush=True)
    
    #------------------#
    print("\tAllocating the calculator ... ", end="")
    model = MACEModel.from_file(file=args.model)
    print("done")
    
    print("\n\tTesing without batching:")
    time_no_batch = np.zeros(args.max_batch)
    for n in range(args.max_batch):
        start_time = time.time()
        model.compute([atoms],raw=True)
        end_time = time.time()
        time_no_batch[n] = end_time - start_time
        # if n > 0 :
        #     time_no_batch[n] += time_no_batch[n-1]
        print(f"\t - {n+1}: {time_no_batch[n]}s")
    time_no_batch  = time_no_batch[args.discard:]
    sizes_no_batch  = np.arange(start=args.discard+1,stop=args.max_batch+1,step=1)
    av_time_no_batch = np.mean(time_no_batch)
    print(f"\n\tAverage time per structure without batching: {av_time_no_batch}s")
    print("\n\tDeleting model ... ",end="")
    del model
    print("done")
    
    #------------------#
    print("\tAllocating the calculator ... ", end="")
    model = MACEModel.from_file(file=args.model)
    print("done")
    
    print("\n\tTesing with batching:")
    time_batch = np.full(args.max_batch,np.nan)
    for n in range(1,args.max_batch+1):
        try :
            start_time = time.time()
            model.compute([atoms]*n,raw=True)
            end_time = time.time()
            time_batch[n-1] = end_time - start_time
            # if n == 50:
            #     raise ValueError("testing error")
            # if n > 0 :
            #     time_no_batch[n] += time_no_batch[n-1]
            print(f"\t - {n}: {time_batch[n-1]}s")
        except:
            break
    time_batch  = time_batch[args.discard:]
    # tot = (~np.isnan(time_batch)).sum()
    sizes  = np.arange(1,args.max_batch+1)
    sizes = sizes[args.discard:]
    sizes = sizes[~np.isnan(time_batch)]
    time_batch = time_batch[~np.isnan(time_batch)]
    time_batch /= sizes
    av_time_batch = np.mean(time_batch)
    print(f"\n\tAverage time per structure with batching: {av_time_batch}s")
    
    factor = av_time_no_batch/av_time_batch
    print(f"\n\tBatching method is {factor:.2f} times faster than the serial one.")
    
    if sizes[-1] != args.max_batch:
        print(f"\tYou can compute at most {sizes[-1]} structures at the same time.\n")
    
    #------------------#
    print("\tSaving plot to file '{:s}' ... ".format(args.output), end="")
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(sizes_no_batch,time_no_batch,color="red",label="serial")
    ax.plot(sizes,time_batch,color="blue",label="batch")
    ax.legend(**legend_options)
    ax.set_xlabel("size")
    ax.set_ylabel("average time [s]")
    ax.grid()
    plt.tight_layout()
    plt.savefig(args.output,dpi=300,bbox_inches="tight")
    print("done")
        
    pass
    

#---------------------------------------#
if __name__ == "__main__":
    main()