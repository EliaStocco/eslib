#!/usr/bin/env python
import numpy as np
from ase.geometry import cellpar_to_cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import flist

#---------------------------------------#
# Description of the script's purpose
description = "Rattle unit cell parameters using gaussian-distributed random numbers."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv, required=True , type=str  , help="input file with atomic structure")
    parser.add_argument("-if" , "--input_format" ,   **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n"  , "--number"       ,   **argv, required=False, type=int  , help="number of output rattled structures for each input structrures (default: %(default)s)", default=10)
    parser.add_argument("-s"  , "--std_devs"     ,   **argv, required=True , type=flist, help="standard deviations of the lattice parameters (a, b, c, α, β, γ) [ang,deg]")
    parser.add_argument("-o"  , "--output"       ,   **argv, required=True , type=str  , help="output file")
    parser.add_argument("-of" , "--output_format",   **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures)
    print("\tn. of structures: {:d}".format(N))
    
    #------------------#
    shape = (N*args.number,6)
    std = np.asarray(args.std_devs)
    random_numbers = np.random.normal(loc=0, scale=std,size=shape)
    print("\n\tGenerated random numbers with shape {}.".format(random_numbers.shape))

    #------------------#
    print("\n\tRattling structures ... ",end="")
    rattled = [None]*(N*args.number)
    k = 0
    for n in range(N):
        original = structures[n].get_cell().cellpar()
        for r in range(args.number):
            cellpar = original + random_numbers[k]
            cell = cellpar_to_cell(cellpar)
            tmp = structures[n].copy()
            tmp.set_cell(cell,scale_atoms=True)
            rattled[k] = tmp
            k += 1
    print("done")
    
    #------------------#
    rattled = AtomicStructures(rattled)
    print("\tn. of rattled structures: {:d}".format(len(rattled)))
                
    #------------------#
    print("\n\tWriting rattled structure to file '{:s}' ... ".format(args.output), end="")
    rattled.to_file(file=args.output,format=args.output_format)
    print("done")    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
