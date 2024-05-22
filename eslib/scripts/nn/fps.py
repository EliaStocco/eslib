#!/usr/bin/env python
from ase.io import read, write
import numpy as np
from skmatter.feature_selection import FPS
from eslib.input import str2bool
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Process atomic structures and select a diverse subset using the Farthest Point Sampling (FPS) algorithm."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"           , type=str     , required=True , **argv, help="input file [au]")
    parser.add_argument("-if" , "--input_format"    , type=str     , required=False, **argv, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-n"  , "--number"          , type=int     , required=True , **argv, help="number of desired structure")
    parser.add_argument("-s"  , "--sort"            , type=str2bool, required=False, **argv, help="whether to sort the indices (default: true)", default=True)
    parser.add_argument("-x"  , "--soap_descriptors", type=str     , required=True , **argv, help="file with the SOAP descriptors")
    parser.add_argument("-oi" , "--output_indices"  , type=str     , required=False, **argv, help="output file with indices of the selected structures (default: 'None')", default=None)
    parser.add_argument("-o"  , "--output"          , type=str     , required=True , **argv, help="output file with the selected structures")
    parser.add_argument("-of" , "--output_format"   , type=str     , required=False, **argv, help="output file format (default: 'None')", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    frames = read(args.input, index=':', format=args.input_format)  #eV
    print("done")

    print("\tReading SOAP descriptors from file '{:s}' ... ".format(args.soap_descriptors),end="")
    if str(args.soap_descriptors).endswith("npy"):
        X = np.load(args.soap_descriptors)
    elif str(args.soap_descriptors).endswith("txt"):
        X = np.loadtxt(args.soap_descriptors)
    print("done")

    #
    print("\tExtracting structures using the FPS algorithm:")
    if args.number == -1:
        args.number = len(frames)
    struct_idx = FPS(n_to_select=args.number, progress_bar = True, initialize = 'random').fit(X.T).selected_idx_
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
        frames_fps = [frames[i] for i in indices]
        write(args.output, frames_fps, format=args.output_format) # fmt)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
