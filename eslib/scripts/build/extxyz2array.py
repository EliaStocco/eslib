#!/usr/bin/env python
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, float_format, eslog
from eslib.input import ilist, str2bool
from eslib.tools import get_files
from eslib.classes.physical_tensor import PhysicalTensor
import concurrent.futures
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Save an 'array' or 'info' from an extxyz file to a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str     , help="input file [extxyz]")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n"  , "--index"        , **argv, required=False , type=str    , help="index (default: %(default)s)", default=":")
    parser.add_argument("-k"  , "--keyword"      , **argv, required=True , type=str     , help="keyword of the info/array")
    parser.add_argument("-s"  , "--shape"        , **argv, required=False, type=ilist   , help="reshape the data (default: %(default)s", default=None)  
    parser.add_argument("-a"  , "--append"       , **argv, required=False, type=str2bool, help="append the trajectories (default: %(default)s)", default=True)
    parser.add_argument("-par", "--parallel"     , **argv, required=False, type=str2bool, help="use parallel algorithm to read files (default: %(default)s)", default=False)
    parser.add_argument("-o"  , "--output"       , **argv, required=False, type=str     , help="output file (default: %(default)s)", default=None)
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str     , help="output format for np.savetxt (default: %(default)s)", default=float_format)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.append:
        
        with eslog(f"Reading atomic structures from file '{args.input}'"):
            atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=args.index)
        
        #------------------#
        # reshape
        print("\t Extracting '{:s}' from the trajectory ... ".format(args.keyword), end="")
        data = atoms.get(args.keyword)  
        print("done")
    
    else:
        #------------------#
        files = get_files(args.input)
        n_files = len(files)
        print("\t Found {:d} files using '{:s}'".format(n_files,args.input))
        for n,file in enumerate(files):
            print("\t\t {:<3d}/{:>3d}: {:s}".format(n,n_files,file))
        print()
        
        #------------------#
        trajectories:List[AtomicStructures] = [None]*n_files
        data:List[np.ndarray] = [None]*n_files
        if args.parallel:
            #------------------#
            with eslog(f"Reading atomic structures from '{args.input}' (parallel)"):
                def read_atomic_structure(file: str) -> AtomicStructures:
                    # with eslog(f"Reading atomic structures from file '{file}'"):
                    return AtomicStructures.from_file(file=file,format=args.input_format,index=args.index)
                # Using ThreadPoolExecutor for I/O-bound tasks
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit tasks to the executor
                    futures = {executor.submit(read_atomic_structure, file): n for n, file in enumerate(files)}

                    for future in concurrent.futures.as_completed(futures):
                        n = futures[future]  # Get the original index
                        try:
                            trajectories[n] = future.result()  # Retrieve the result
                        except Exception as e:
                            print(f"Error reading file {files[n]}: {e}")
        else:
            #------------------#
            for n,file in enumerate(files):
                with eslog(f"Reading atomic structures from file '{file}'"):
                    trajectories[n] = AtomicStructures.from_file(file=file,format=args.input_format,index=args.index)
            
            
            for n,(traj,file) in enumerate(zip(trajectories,files)):
                with eslog(f"Extracting '{args.keyword}' from the trajectory of file '{file}'"):
                    data[n] = traj.get(args.keyword)
                    
            data = np.asarray(data)

    print("\t '{:s}' shape: ".format(args.keyword),data.shape)

    print("\n\t data type: ",data.dtype)

    #------------------#
    if np.issubdtype(data.dtype, np.str_):
        print("\t Data contains strings: -of/--output_format will be set to '%s'" % "%s")
        args.output_format = "%s"

    if args.shape is not None:
        shape = tuple(args.shape)
        print("\t Reshaping data to ",shape," ... ",end="")
        data = data.reshape(shape)
        print("done")

    #------------------#
    print("\n\t Converting data into PhysicalTensor ... ", end="")
    data = PhysicalTensor(data)
    print("done")

    #------------------#
    if args.output is None:
        file = "{:s}.txt".format(args.keyword)
    else:
        file = str(args.output)

    print("\t Storing '{:s}' to file '{:s}' ... ".format(args.keyword,file), end="")
    data.to_file(file=file,fmt=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()

