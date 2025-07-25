#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from skmatter.feature_selection import FPS

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Select a diverse subset of structures using the Farthest Point Sampling (FPS) algorithm."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"           , type=str     , required=True , **argv, help="input file [au]")
    parser.add_argument("-if" , "--input_format"    , type=str     , required=False, **argv, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n"  , "--number"          , type=int     , required=False, **argv, help="number of desired structure (default: %(default)s", default=-1)
    parser.add_argument("-s"  , "--sort"            , type=str2bool, required=False, **argv, help="whether to sort the indices (default: %(default)s)", default=False)
    parser.add_argument("-x"  , "--soap_descriptors", type=str     , required=False, **argv, help="file with the SOAP descriptors (default: %(default)s)", default="soap.npy")
    parser.add_argument("-oi" , "--output_indices"  , type=str     , required=False, **argv, help="output file with indices of the selected structures (default: %(default)s)", default="indices.txt")
    parser.add_argument("-o"  , "--output"          , type=str     , required=False, **argv, help="output file with the selected structures (default: %(default)s)", default="fps.extxyz")
    parser.add_argument("-of" , "--output_format"   , type=str     , required=False, **argv, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    if args.number == -1 and args.sort:
        raise ValueError("You can not sort the indices and consider all the structures. It's just meaningless.") 
    
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    frames = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")

    print("\tReading SOAP descriptors from file '{:s}' ... ".format(args.soap_descriptors),end="")
    if str(args.soap_descriptors).endswith("npy"):
        X = np.load(args.soap_descriptors)
    elif str(args.soap_descriptors).endswith("txt"):
        X = np.loadtxt(args.soap_descriptors)
    print("done")
    
    assert X.shape[0] == len(frames), f"SOAP descriptors have shape {X.shape} while n. of structures is {len(frames)}."
    
    if args.number == -1:
        args.number = len(frames)
    if args.number > len(frames):
        args.number = len(frames)
    
    print("\tExtracting structures using the FPS algorithm:")
    # mean over structures of each descriptor
    mean = X.mean(axis=0,keepdims=True) # axis = 0:  structures, axis = 1: descriptors
    delta = X - mean
    # norm over descriptors of each structure
    norm = np.linalg.norm(delta,axis=1)
    # initialize the FPS algorithm with the structure with the largest norm
    initial = np.argmax(norm)
    struct_idx = FPS(n_to_select=args.number, progress_bar = True, initialize=initial).fit(X.T).selected_idx_
    X_fps = X[struct_idx]

    print("\n\tFPS selected indices: {:d}".format(struct_idx.shape[0]))
    print(f"\tOriginal: {X.shape} ---> FPS: {X_fps.shape}")

    indices = np.asarray([ int(i) for i in struct_idx],dtype=int)

    if args.sort:
        print("\n\tSorting indices ... ",end="")
        indices = np.sort(indices)
        print("done")

    

    # Saving the fps selected structure
    if args.output_indices :
        print("\n\tSaving indices of selected structures to file '{:s}' ... ".format(args.output_indices),end="")
        np.savetxt(args.output_indices,indices,fmt='%d')
        print("done")

    # Saving the fps selected structure
    if args.output is not None :
        print("\n\tSaving FPS selected structures to file '{:s}' ... ".format(args.output),end="")
        frames_fps = frames.subsample(indices) # [frames[i] for i in indices]
        frames_fps.to_file(file=args.output, format=args.output_format) # fmt)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
