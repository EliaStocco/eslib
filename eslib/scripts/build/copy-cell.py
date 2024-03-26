#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from eslib.classes.trajectory import trajectory as Trajectory
from eslib.formatting import esfmt
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Copy the cell from one file to a trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with the atomic structures to be modified")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-c" , "--cell"          , **argv, required=True , type=str, help="file with the atomic structure with the cell to be copied")
    parser.add_argument("-cf", "--cell_format"   , **argv, required=False, type=str, help="input cell file format (default: 'None')" , default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file with the modified atomic structures")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: 'None')", default=None)
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # cell
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.cell), end="")
    periodic:Atoms = list(Trajectory(args.cell,format=args.cell_format,index=0))[0]
    cell = periodic.get_cell()
    pbc = periodic.get_pbc()
    print("done")

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory:List[Atoms] = list(Trajectory(args.input,format=args.input_format,index=":"))
    print("done")
    
    #------------------#
    # replace
    print("\tReplace cell ... ", end="")
    for atoms in trajectory:
        atoms.set_calculator(None)
        atoms.cell = cell
        atoms.pbc = pbc
    print("done")

    #------------------#
    # output

    print("\n\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=trajectory,filename=args.output, format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))    

#---------------------------------------#
if __name__ == "__main__":
    main()