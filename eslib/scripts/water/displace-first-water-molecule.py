#!/usr/bin/env python
import numpy as np
from typing import List
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog

#---------------------------------------#
# Description of the script's purpose
description = "Displace the first water molecule."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="input file [extxyz]")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-d" , "--displacement" , **argv, required=False, type=float, help="displacement [ang] (default: %(default)s)", default=0.01)
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str  , help="molecule name (default: %(default)s)", default="molecule")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading atomic structures from file '{args.input}'"):
        trajectory:List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    Ns = len(trajectory)
    print("\t Number atomic structures: ",Ns)
    
    #------------------#
    with eslog("Displacing atoms"):
        out_traj = [None]*(9*Ns+Ns)
        k = 0
        for n in range(Ns):
            
            out_traj[k] = trajectory[n].copy()
            k += 1
            
            mol = trajectory[n].arrays[args.molecule]
            ii = mol == 0
            pos_original = np.asarray(trajectory[n].get_positions()[ii]).flatten().copy()
            
            for xyz in range(9):
                pos = pos_original.copy()
                atoms = trajectory[n].copy()
                pos[xyz] += args.displacement
                
                atoms.positions[ii] = pos.reshape((3,3))
                
                out_traj[k] = atoms.copy()
                
                k += 1
         
    displaced = AtomicStructures(out_traj)       
    
    #------------------#
    print(f"\t Writing atomic structures with {n} atoms to file '{args.output}' ... ", end="")
    displaced.to_file(file=args.output,format=args.output_format)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()


