#!/usr/bin/env python
import numpy as np
from ase.io import write, read
from copy import copy
from eslib.tools import cart2lattice, lattice2cart
from eslib.tools import frac2cart, cart2frac
from eslib.input import flist, str2bool
from eslib.formatting import esfmt, warning

#---------------------------------------#
description = "Fix the dipole jumps and shift the values of some multitples of the dipole quantum."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"   , type=str     , **argv, required=True , help="input 'extxyz' file")
    parser.add_argument("-n" , "--name"   , type=str     , **argv, required=False, help="name of the info to be handled (default: 'dipole')", default="dipole")
    parser.add_argument("-f", "--fix"     , type=str2bool, **argv, required=False, help="whether to fix only the jumps, without shifting (default: false)", default=False)
    parser.add_argument("-j", "--jumps"   , type=str     , **argv, required=False, help="output txt file with jumps indeces (default: 'None')", default=None)
    parser.add_argument("-a", "--average_shift", type=str2bool   , **argv, required=False, help="whether to shift the dipole quanta by their average value (default: true)", default=True)
    parser.add_argument("-s", "--shift"   , type=flist   , **argv, required=False, help="additional (negative) shift (default: [0,0,0])", default=None)
    parser.add_argument("-q", "--quanta"  , type=str   , **argv, required=False, help="keyword for the dipole quanta (default: 'quanta')", default='quanta')
    parser.add_argument("-o", "--output"  , type=str     , **argv, required=True , help="output 'extxyz' file")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    ###
    # read the MD trajectory from file
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format='extxyz',index=":")
    print("done")


    print("\n\tConverting '{:s}' from cartesian to lattice coordinates ... ".format(args.name), end="")
    N = len(atoms)
    phases = np.full((N,3),np.nan)
    for n in range(N):
        # for some strange reason I need the following line .... ???
        atoms[n].calc = None # atoms[n].set_calculator(None)
        phases[n,:] = cart2frac(cell=atoms[n].get_cell(),v=atoms[n].info[args.name])
    print("done\n")

    if args.fix :
        print("\tFixing the '{:s}' jumps ... ".format(args.name), end="")
        old = phases.copy()
        for i in range(3):
            phases[:,i] = np.unwrap(phases[:,i],period=1)
        print("done")

        index = np.where(np.diff(old-phases,axis=0))[0]
        print("\n\tFound {:d} jumps".format(len(index)))
        if args.jumps is not None:
            print("\tSaving the indices of the jumps to file '{:s}' ... ".format(args.jumps), end="")
            np.savetxt(args.jumps,index)
            print("done")
        print()

    shift = np.zeros(3)
    if args.average_shift:
        shift = np.asarray([ i.round(0) for i in phases.mean(axis=0) ]).astype(float)
        print("\tThe dipole quanta will be shifted by the average value: ",shift)
        
    if args.shift is not None:        
        print("\tUser-defined shift of the dipole quanta: ",args.shift)
        shift += args.shift

    print("\tShifting the dipoles quanta by ",shift, " ... ",end="")
    for i in range(3):
        phases[:,i] -= shift[i]
    print("done")


    print("\tConverting dipoles from lattice to cartesian coordinates ... ", end="")
    for n in range(N):
        atoms[n].info[args.name] = frac2cart(cell=atoms[n].get_cell(),v=phases[n,:]).reshape((3,))
    print("done")

    if args.quanta is not None:
        if args.quanta in atoms[0].info:
            print("\t{:s}: info '{:s}' will be overwritten.".format(warning,args.quanta))
        print("\tAdding dipole quanta as info 'quanta' to atomic structures ... ", end="")
        for n in range(N):
            atoms[n].info[args.quanta] = phases[n,:]
        print("done")

    ###
    # writing
    print("\n\tWriting output to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, atoms, format="extxyz")
        print("done")
    except Exception as e:
        print(f"\n\tError: {e}")

if __name__ == "__main__":
    main()
