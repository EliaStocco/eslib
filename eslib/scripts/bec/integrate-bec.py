#!/usr/bin/env python
from typing import List

import numpy as np
from ase import Atoms
from ase.io import write

from classes import structure
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import dec_format, esfmt, warning
from eslib.input import str2bool
from eslib.tools import convert
from eslib.tools import cart2frac, frac2cart

#---------------------------------------#
# Description of the script's purpose
description = "Integrate Born Charges."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="input file")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-b" , "--bec"          , **argv, required=False, type=str, help="keyword for Born Charges (default: %(default)s)", default="BEC")
    parser.add_argument("-d" , "--dipole"       , **argv, required=False, type=str, help="keyword for dipole (default: %(default)s)", default="dipole")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)" , default=None)
    return parser
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(trajectory)
    Na = trajectory.num_atoms()
    print("\tn. of structures: ",N)
    print("\tn. of atoms: ",Na)
    
    #------------------#
    print("\n\tExtracing positions ... ",end="")
    pos = trajectory.get("positions")
    print("done")
    print("\tpos.shape: ",pos.shape)
    
    #------------------#
    print("\n\tExtracing Born Charges ... ",end="")
    bec = trajectory.get(args.bec)
    print("done")
    print("\tbec.shape: ",bec.shape)
    bec = bec.reshape((bec.shape[0],bec.shape[1],3,3))
    
    #------------------#
    cell = trajectory[0].get_cell()
    symbols = trajectory[0].get_chemical_symbols()
    delta = pos[0] - pos[-1]
    ii = np.any(delta!=0,axis=1)
    atoms = np.asarray(symbols)[ii]
    print("\n\tDisplaced atoms: ", atoms)
    if len(atoms) == 0:
        raise ValueError("Please unfold your trajectory first.")
    elif len(atoms) > 1:
        assert all([a ==  atoms[0] for a in atoms]), "Displaced atoms of different species"
        
    direction = delta[ii][0]
    frac = np.round(cart2frac(cell,direction),2)
    print("\tDisplacement direction: ", frac.tolist())
    
    #------------------#
    print("\n\tComputing dipole ... ",end="")
    dR = np.diff(pos,axis=0)
    assert np.allclose(dR,dR[0],atol=1e-4), "something weird"
    
    dR = np.zeros(pos.shape)
    dR[:,ii,:] = - cell.T@np.asarray(frac)/(N-1)
    
    d_mu = np.einsum("ijab,ija->ib",bec,dR)
    dipole = np.cumsum(d_mu,axis=0)
    dipole -= dipole[0]
    
    trajectory.set(args.dipole,dipole,"info")
    print("done")
    
    #-------------------#
    print("\n\tWriting structures to file '{:s}' ... ".format(args.output), end="")
    trajectory.to_file(file=args.output,format=args.output_format)
    print("done")
    

    pass

#---------------------------------------#
if __name__ == "__main__":
    main()