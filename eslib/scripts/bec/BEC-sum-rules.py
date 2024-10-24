#!/usr/bin/env python
import json

import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.bec import bec
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Check that the BECs satisfy the sum rules."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--pos_file"       , **argv, required=True , type=str, help="atomic structures file [extxyz]")
    parser.add_argument("-if", "--pos_format", **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-z" , "--BEC_file"       , **argv, required=True , type=str, help="input file with the BECs")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.pos_file), end="")
    trajectory = AtomicStructures.from_file(file=args.pos_file,format=args.pos_format,index=":")
    print("done")
    print("\tn. of atomic structures: {:d}".format(len(trajectory)))

    #------------------#
    print("\n\tReading the BECs from file '{:s}' ... ".format(args.BEC_file), end="")
    Z = bec.from_file(file=args.BEC_file)
    print("done")
    print("\tBECs.shape: {}".format(Z.shape))

    assert len(Z) == len(trajectory), "The number of BECs does not match the number of atomic structures"

    #------------------#
    print("\n\tExpanding BECs into atomic BECS ... ", end="")
    Z = Z.expand_with_atoms()
    print("done")
    print("\tBECs.shape: {}".format(Z.shape))

    #------------------#
    print("\n\tChecking translational sum rule ... ", end="")
    aerr = np.full(len(Z),np.nan) # abs. error
    rerr = np.full(len(Z),np.nan) # rel. error
    for n,z in enumerate(Z):
        aerr[n] = float(np.mean(np.sum(z,axis=0)))
        norm = np.linalg.norm(z)
        assert norm == np.sqrt(np.sum(np.asarray(z).flatten()**2)), "coding error"
        rerr[n] = float(aerr[n] / norm)
    print("done")

    print("\t - absolute error: {:>4e}".format(aerr.mean()))
    print("\t - relative error: {:>4e}".format(rerr.mean()))


    #------------------#
    print("\n\tChecking rotational sum rule ... ", end="")
    aerr = np.full(len(Z),np.nan) # abs. error
    rerr = np.full(len(Z),np.nan) # rel. error
    positions = trajectory.get("positions")
    for n,z in enumerate(Z.to_numpy()):
        # z.shape   = (80,3,3)
        # pos.shape = (80,3)
        pos = positions[n]
        tot = np.cross(z,pos[:,:,np.newaxis],axisa=1,axisb=1).sum(axis=0)
        # cross = np.full((len(z),3,3),np.nan)
        for i,(zi,ri) in enumerate(zip(z,pos)):
            # zi.shape = (3,3)
            # ri.shape = (3)
            # cross.shape = (3,3)
            cross[i] = np.cross(zi,ri,axisa=0)
        cross = cross.sum(axis=0)
        aerr[n] = float(np.mean(np.sum(z,axis=0)))
        norm = np.linalg.norm(z)
        assert norm == np.sqrt(np.sum(np.asarray(z).flatten()**2)), "coding error"
        rerr[n] = float(aerr[n] / norm)
    print("done")

    print("\t - absolute error: {:>4e}".format(aerr.mean()))
    print("\t - relative error: {:>4e}".format(rerr.mean()))


    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()