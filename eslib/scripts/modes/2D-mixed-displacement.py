#!/usr/bin/env python
import json
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from ase.io import read

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt

#---------------------------------------#
# example of file for --displacements
# {
#   "N-AB" : 10, // number of intermediate structures between A and B 
#   "N-M" : 5,   // number of structures along the other mode (positive axis only) --> they will be (N_M)+(N_M-1)
#   "M-d" : 2.1, // maximum displacement in atomic_unit along the other mode
#   "MA" : 8,    // index of the normal mode for structure A
#   "MB" : 8     // index of the normal mode for structure B
# }

#---------------------------------------#
# Description of the script's purpose
description = "Displace an atomic structure along two normal modes, with one of these that is 'mixed' between the intial and final structure."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a" , "--structure_A"  , type=str, required=True , **argv, help="extxyz file with the structure A [a.u.]")
    parser.add_argument("-b" , "--structure_B"  , type=str, required=True , **argv, help="fextxyz file with the structure A [a.u.]")
    parser.add_argument("-ma", "--modes_A"      , type=str, required=True , **argv, help="pickle file with the normal modes of the structure A")
    parser.add_argument("-mb", "--modes_B"      , type=str, required=True , **argv, help="pickle file with the normal modes of the structure B")
    parser.add_argument("-i" , "--instructions" , type=str, required=True , **argv, help="JSON file with the instructions")
    parser.add_argument("-o" , "--output"       , type=str, required=False, **argv, help="extxyz output file [a.u.] (default: %(default)s)", default='displaced-structures.extxyz')
    parser.add_argument("-oi", "--output_info"  , type=str, required=False, **argv, help="csv output file (default: %(default)s)", default='info.csv')
    parser.add_argument("-od" , "--out_dis"     , type=str, required=False, **argv, help="*.txt output file with the displacements (default: %(default)s)", default='displacement.txt')
    return parser

@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from file '{:s}' ... ".format(args.structure_A), end="")
    structure_A = read(args.structure_A,index=0)
    print("done")

    #------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    structure_B = read(args.structure_B,index=0)
    print("done")

    #------------------#
    print("\tReading normal modes of A from file '{:s}' ... ".format(args.modes_A), end="")
    modes_A = NormalModes.from_pickle(args.modes_A)
    print("done")

    #------------------#
    print("\tReading normal modes of B from file '{:s}' ... ".format(args.modes_B), end="")
    modes_B = NormalModes.from_pickle(args.modes_B)
    print("done")

    #------------------#
    print("\tReading the instructions from file '{:s}' ... ".format(args.instructions), end="")
    with open(args.instructions, 'r') as f:
        instructions = json.load(f)
    print("done")

    #------------------#
    print("\tReading the instructions from file '{:s}' ... ".format(args.instructions), end="")
    with open(args.instructions, 'r') as f:
        instructions = json.load(f)
    print("done")

    #------------------#
    print("\n\tPreparing the calculation of the displaced structures ... ", end="")
    A2B_mode = np.asarray(structure_B.get_positions() - structure_A.get_positions()).flatten()
    A2B_distance = np.sqrt(np.sum(np.square(A2B_mode)))
    A2B_mode /= A2B_distance
    # test 
    assert np.allclose( structure_B.get_positions() , structure_A.get_positions() + A2B_distance* A2B_mode.reshape((-1,3)))

    #------------------#
    MixedModes = NormalModes(Nmodes=2,Ndof=modes_A.Ndof,ref=structure_A)

    #------------------#
    N_AB = instructions['N-AB']+2
    N_M = instructions['N-M']+(instructions['N-M']-1)
    Ndispl = N_AB * N_M
    displaced_structures = [None]*(Ndispl)
    displ_information = pd.DataFrame(columns=['A-B [bohr]','M [bohr]',
                                              'A-B [index]','M [index]'],index=np.arange(Ndispl))
    
    Ma = modes_A.mode.isel(mode=instructions['MA'])
    Mb = modes_B.mode.isel(mode=instructions['MB'])
    print("done")

    #------------------#
    print("\tn. of displaced structures:",Ndispl)
    print("\tA-B distance: {:4f} bohr".format(A2B_distance))

    #------------------#
    a = np.asarray(Ma)
    b = np.asarray(Mb)
    prod = a @ b
    sign = np.sign(prod).astype(int)
    print("\n\tScalar product between modes: {:.4f}".format(prod))
    print("\tThe second mode will be multiplied by {:d}".format(sign))

    #------------------#
    print("\n\tDisplacing the structures ... ", end="")
    n = 0 
    for ab in range(N_AB):
        for m in range(-instructions['N-M']+1, instructions['N-M']):
            # ic(m,ab)

            # set up the output DataFrame
            displ_information.at[n,'A-B [index]'] = ab 
            displ_information.at[n,'A-B [bohr]'] = ab * A2B_distance/(N_AB-1)
            displ_information.at[n,'M [index]'] = m 
            displ_information.at[n,'M [bohr]'] = m * instructions['M-d']/(instructions['N-M']-1)
            # displ_information.at[n,'MA [bohr]'] = float(1-ab)/(N_AB-1)# *displ_information.at[n,'M [bohr]']
            # displ_information.at[n,'MB [bohr]'] = float(ab)/(N_AB-1)# *displ_information.at[n,'M [bohr]']

            # set up the normal modes
            modes = np.zeros((MixedModes.Ndof,2))
            modes[:,0] = A2B_mode
            modes[:,1] = float(1-ab)/(N_AB-1)*Ma + sign * float(ab)/(N_AB-1)*Mb
            MixedModes.set_modes(modes)
            # ic(modes.T)
            # if np.any(np.isnan(modes)):
            #     pass

            # set up the displacements
            bohr_disp = xr.DataArray(np.asarray([displ_information.at[n,'A-B [bohr]'],
                                                 displ_information.at[n,'M [bohr]']]), dims=("mode"))
            displaced_structures[n] = MixedModes.nmd2cp(bohr_disp)
            n += 1
    print("done")

    #------------------#
    print("\n\tWriting displaced structures to file '{:s}' ... ".format(args.output), end="")
    displaced_structures = AtomicStructures(displaced_structures)
    displaced_structures.to_file(file=args.output)
    print("done")

    #------------------#
    print("\n\tWriting information to file '{:s}' ... ".format(args.output_info), end="")
    displ_information.to_csv(args.output_info,index=False)
    print("done")

    #------------------#
    print("\n\tWriting displacements to file '{:s}' ... ".format(args.out_dis), end="")
    # tmp = pd.DataFrame(columns=["mode","start","end","N"],index=[0,1])

    # tmp.at[0,"mode"]  = 0
    # tmp.at[0,"start"] = 0
    # tmp.at[0,"end"]   = 1
    # tmp.at[0,"N"]     = args.shape[0]-1

    # tmp.at[1,"mode"]  = 1
    # tmp.at[1,"start"] = 0
    # tmp.at[1,"end"]   = 1
    # tmp.at[1,"N"]     = args.shape[1]-1

    # #------------------#
    # print("\tPreparing displacement along single modes:")
    # lists = [None]*len(tmp)
    # for n,row in tmp.iterrows():
    #     print("\t\tmode {:3d}: ".format(int(row['mode'])),end="")
    #     lists[n]= np.linspace(row['start'],row['end'],int(row['N']+1),endpoint=True)
    #     print(lists[n])
    sud_df = displ_information[ ['M [bohr]','A-B [bohr]'] ]
    displacements = np.asarray(sud_df)
    print(displacements.shape)
    np.savetxt(args.out_dis,displacements,fmt='%10.4f')
    print("done") 
    
    return 0     

if __name__ == "__main__":
    main()


# { 
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/modes/2D-mixed-displacement.py",
#             "cwd" : "/home/stoccoel/google-personal/simulations/LiNbO3/skew-1x1x1/displ",
#             "console": "integratedTerminal",
#             "args" : ["-a", "std.au.extxyz", "-b", "inv.au.extxyz", "-ma", "vib-std.pickle", "-mb", "vib-inv.pickle", "-i", "instructions.json"],
#             "justMyCode": false,
#         }
#     ]
# }