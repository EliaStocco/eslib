#!/usr/bin/env python
import numpy as np
from ase.io import write, read
from copy import copy
from eslib.tools import cart2lattice, lattice2cart, frac2cart
from eslib.input import flist, str2bool
from eslib.formatting import esfmt

#---------------------------------------#
description = "Fix the dipole jumps and shift the values of some multitples of the dipole quantum.\n"

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
    parser.add_argument("-s", "--shift"   , type=flist   , **argv, required=False, help="additional vector to be added to the output file (default: [0,0,0])", default=None)
    parser.add_argument("-o", "--output"  , type=str     , **argv, required=True , help="output 'extxyz' file")
    return parser.parse_args()

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
    lenght = np.full((N,3),np.nan)
    # old = np.full((N,3),np.nan)
    for n in range(N):
        atoms[n].set_calculator(None)
        cell = np.asarray(atoms[n].cell.array).T
        lenght[n,:] = np.linalg.norm(cell,axis=0)
        R = cart2lattice(cell)
        dipole = R @ atoms[n].info[args.name]
        phases[n,:] = dipole / lenght[n,:]
    print("done")

    if args.fix :
        print("\tFixing the '{:s}' jumps ... ".format(args.name), end="")
        old = phases.copy()
        for i in range(3):
            phases[:,i] = np.unwrap(phases[:,i],period=1)
        print("done")

    # if args.shift is not None:
    #     shift = np.asarray([ int(i) for i in phases.mean(axis=0) ]).astype(float)
    #     print("\tShifting the dipoles by the following quantum (on top of the global mean): ",args.shift," ... ", end="")
    #     shift += args.shift
    #     for i in range(3):
    #         phases[:,i] -= shift[i]
    #     print("done")

    if args.shift is not None :
        shift = np.zeros(3)
        if args.average_shift:
            shift = np.asarray([ int(i) for i in phases.mean(axis=0) ]).astype(float)
            print("\tThe dipole quanta will be shifted by the average value: ",shift)
        if args.shift is not None:        
            print("\tUser-defined shift of the dipole quanta: ",args.shift)
            shift += args.shift
        print("\tShifting the dipoles quanta by ",shift, " ... ",end="")
        for i in range(3):
            phases[:,i] -= shift[i]
        print("done")

    if args.fix or args.shift is not None:
        print("\tConverting dipoles from lattice to cartesian coordinates ... ", end="")
        for n in range(N):
            # cell = np.asarray(atoms[n].cell.array).T
            # R = lattice2cart(cell)
            # atoms[n].info[args.name] = R @ ( phases[n,:] * lenght[n,:] )

            # The previous code lines should be the same as the following:
            atoms[n].info[args.name] = frac2cart(cell=atoms[n].get_cell(),v=phases[n,:])
        print("done")
        

    if args.fix:
        index = np.where(np.diff(old-phases,axis=0))[0]
        print("\n\tFound {:d} jumps".format(len(index)))
        if args.jumps is not None:
            print("\tSaving the indices of the jumps to file '{:s}' ... ".format(args.jumps), end="")
            np.savetxt(args.jumps,index)
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
