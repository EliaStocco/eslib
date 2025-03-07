#!/usr/bin/env python
import argparse
import pickle

import xarray as xr

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt, warning
from eslib.output import output_folder
from eslib.units import remove_unit

# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
# Description of the script's purpose
description = "Check thermalization of a trajectory by looking at the phonon modes energies."

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-t" ,  "--trajectory"   , type=str, **argv, help="input extxyz file [a.u.] (default: %(default)s)", default="trajectory.extxyz")
    parser.add_argument("-nm",  "--normal_modes" , type=str, **argv, help="normal modes file generated by 'iPI2NormalModes.py' (default: %(default)s)", default="normal-modes.pickle")
    parser.add_argument("-o" ,  "--output"       , type=str, **argv, help="output file (default: %(default)s)", default="thermalization.pdf")
    return parser

@esfmt(prepare_args,description)
def main(args):

    print("\tReading trajectory from file '{:s}' ... ".format(args.trajectory), end="")
    trajectory = AtomicStructures.from_file(file=args.trajectory)
    print("done")

    #---------------------------------------#
    # read phonon modes ('phonon-modes.pickle')
    print("\tReading phonon modes from file '{:s}' ... ".format(args.normal_modes), end="")
    with open(args.normal_modes,'rb') as f:
        nm = pickle.load(f)
    print("done")

    if type(nm) != NormalModes:
        raise TypeError("Loaded object is of wrong type, it should be a 'NormalModes' object")

    #---------------------------------------#
    # project on phonon modes
    print("\n\tProjecting the trajectory:")
    results = nm.project(trajectory,warning)
    print("done")
   
        
    pass

if __name__ == "__main__":
    main()