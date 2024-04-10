#!/usr/bin/env python
import json
import numpy as np
from eslib.classes.dipole import DipolePartialCharges
from eslib.classes.trajectory import AtomicStructures, astype
from eslib.formatting import esfmt, warning
from eslib.show import show_dict
from eslib.tools import is_integer
from eslib.classes.bec import bec

#---------------------------------------#
# Description of the script's purpose
description = "Provide a summary of the informations related to the BECs."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: 'None')", default=None)
    parser.add_argument("-n" , "--name"        , **argv, required=False, type=str, help="name of the BEC (default: 'BEC')", default='BEC')
    parser.add_argument("-s" , "--summary"     , **argv, required=False, type=str, help="JSON file with a summary (default: None)", default=None)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="*.txt output file (default: 'bec.txt')", default='bec.txt')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\n\tReading the atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory:AtomicStructures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of atomic structures: {:d}".format(len(trajectory)))

    if np.all( [ trajectory.is_there(f'{args.name}{xyz}') for xyz in ["x","y","z"] ] ):
        Zx = trajectory.get_array(f'{args.name}x')
        Zy = trajectory.get_array(f'{args.name}y')
        Zz = trajectory.get_array(f'{args.name}z')
        Z = bec.from_components(Zx,Zy,Zz)
    else:   
        Z = trajectory.get_array(args.name)
        Z = bec.from_numpy(Z)

    # Z = Z.expand_with_atoms()
    
    #------------------#
    print("\n\tWriting BECs to file '{:s}' ... ".format(args.output), end="")
    Z.to_file(args.output)
    print("done")

    if args.summary is not None:
        print("\n\tWriting BECs summary to file '{:s}' ... ".format(args.output), end="")
        data = Z.summary()
        with open(args.summary,"w") as ffile:
            json.dump(obj=data,fp=ffile,indent=4)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()