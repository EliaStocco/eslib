#!/usr/bin/env python
import numpy as np
from skmatter.feature_selection import FPS
from eslib.input import str2bool
from eslib.formatting import esfmt
from python_tsp import exact
from python_tsp import heuristics
from eslib.classes.atomic_structures import AtomicStructures
from eslib.functions import suppress_output
 
# git@github.com:fillipe-gsm/python-tsp.git
methods = {
    "branch" : exact.solve_tsp_branch_and_bound,
    "brute" : exact.solve_tsp_brute_force,
    "dynamic" : exact.solve_tsp_dynamic_programming,
    "kernighan" : heuristics.solve_tsp_lin_kernighan,
    "local" : heuristics.solve_tsp_local_search,
    "record" : heuristics.solve_tsp_record_to_record,
    "annealing" : heuristics.solve_tsp_simulated_annealing
}

#---------------------------------------#
# Description of the script's purpose
description = "Sort structures according to the a SOAP descriptors-based distance."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"           , type=str     , required=True , **argv, help="input file [au]")
    parser.add_argument("-if" , "--input_format"    , type=str     , required=False, **argv, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-d"  , "--distance"        , type=str     , required=True , **argv, help="file with the distance matrix")
    parser.add_argument("-m"  , "--method"          , type=str     , required=False, **argv, help="sorting method (default: %(default)s)", default='local', choices=list(methods.keys()))
    parser.add_argument("-l"  , "--log_file"        , type=str     , required=False, **argv, help="log file (default: %(default)s)", default="log.out")
    parser.add_argument("-N"  , "--max_N"           , type=int     , required=False, **argv, help="maximum number of iterations (default: %(default)s)", default=10)
    parser.add_argument("-oi" , "--output_indices"  , type=str     , required=False, **argv, help="output file with indices of the selected structures (default: %(default)s)", default=None)
    # parser.add_argument("-oi" , "--output_indices"  , type=str     , required=False, **argv, help="output file with indices of the selected structures (default: %(default)s)", default=None)
    parser.add_argument("-o"  , "--output"          , type=str     , required=True , **argv, help="output file with the selected structures")
    parser.add_argument("-of" , "--output_format"   , type=str     , required=False, **argv, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    print("\n\tReading atomic structures from file '{:s}' ... ".format(args.input),end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format) 
    print("done")

    #------------------#
    print("\n\tReading distance matrix from file '{:s}' ... ".format(args.distance),end="")
    if str(args.distance).endswith("npy"):
        D = np.load(args.distance)
    elif str(args.distance).endswith("txt"):
        D = np.loadtxt(args.distance)
    print("done")
    print("\tDistance matrix shape: ",D.shape)

    #------------------#
    print("\n\tNormalizing distance matrix ... ",end="")
    D /= np.linalg.norm(D)
    print("done")

    #------------------#
    print("\n\tSorting structures using '{:s}' algorithm ... ".format(args.method),end="")
    method:callable = methods[args.method]
    max_N = None if args.max_N <= 0 else args.max_N
    with suppress_output():
        try:
            permutation, distance = method(D,log_file=args.log_file,verbose=True,max_iterations=max_N)
        except:
            permutation, distance = method(D,log_file=args.log_file,verbose=True,max_processing_time=max_N)
    print("done")

    return 0   

#---------------------------------------#
if __name__ == "__main__":
    main()
