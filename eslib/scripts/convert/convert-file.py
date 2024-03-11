#!/usr/bin/env python
from ase.io import read, write
from ase.cell import Cell
# from elia.functions import str2bool, suppress_output, convert
from eslib.classes.trajectory import trajectory
import numpy as np
import contextlib
import sys
import os
from eslib.formatting import esfmt, warning, error
from eslib.input import str2bool
from eslib.functions import suppress_output
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Convert the format and unit of a file using 'ASE'"
keywords = "It's up to you to modify the required keywords."

# Attention:
# If the parser used in ASE automatically modify the unit of the cell and/or positions,
# then you should add this file format to the list at line 55 so that the user will be warned.

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv,required=True , type=str     , help="input file")
    parser.add_argument("-if" , "--input_format" ,   **argv,required=False, type=str     , help="input file format (default: 'None')" , default=None)
    parser.add_argument("-rr" , "--remove_replicas", **argv,required=False, type=str2bool, help='whether to remove replicas (default: false)', default=False)
    parser.add_argument("-pbc", "--pbc"          ,   **argv,required=False, type=str2bool, help="whether pbc should be removed, enforced, or nothig (default: 'None')", default=None)
    parser.add_argument("-iu" , "--input_unit"   ,   **argv,required=False, type=str     , help="input positions unit (default: atomic_unit)"  , default=None)
    parser.add_argument("-iuc", "--input_unit_cell", **argv,required=False, type=str, help="input cell unit (default: atomic_unit)"  , default=None)
    parser.add_argument("-ou" , "--output_unit" ,    **argv,required=False, type=str     , help="output unit (default: atomic_unit)", default=None)
    parser.add_argument("-s"  , "--scaled"      ,    **argv,required=False, type=str2bool, help="whether to output the scaled positions (default: False)", default=False)
    parser.add_argument("-r"  , "--rotate" ,         **argv,required=False, type=str2bool     , help="whether to rotate the cell s.t. to be compatible with i-PI (default: False)", default=False)
    parser.add_argument("-n"  , "--index" ,          **argv,required=False, type=lambda x: int(x) if x.isdigit() else x     , help="index to be read from input file (default: ':')", default=':')
    parser.add_argument("-o"  , "--output"       ,   **argv,required=True , type=str     , help="output file")
    parser.add_argument("-of" , "--output_format",   **argv,required=False,type=str     , help="output file format (default: 'None')", default=None)
    parser.add_argument("-f"  , "--folder"       ,   **argv,required=False,type=str     , help="folder of the output files if each structure has to be saved in a different file (default: None)", default=None)
    # Parse the command-line arguments
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.input_format is None:
        print("\tDeducing input file format: ", end="")
        from ase.io.formats import filetype
        args.input_format = filetype(args.input, read=True)
        print(args.input_format)
    if args.output_format is None:
        print("\tDeducing output file format: ", end="")
        from ase.io.formats import filetype
        args.output_format = filetype(args.output, read=False)
        print(args.output_format)

    if args.input_format is None:
        raise ValueError("coding error: 'args.input_format' is None.")
    
    if args.output_format is None:
        raise ValueError("coding error: 'args.output_format' is None.")
        
    #------------------#
    print("\tReading data from input file '{:s}' ... ".format(args.input), end="")
    with suppress_output():
        # Try to determine the format by checking each supported format
        atoms = trajectory(args.input,format=args.input_format,index=args.index,pbc=args.pbc,remove_replicas=args.remove_replicas)
        atoms = list(atoms)
    print("done")
    print("\tn. of atomic structures: {:d}".format(len(atoms)))

    #------------------#
    if args.input_format in ["espresso-in","espresso-out"] and args.output_format in ["espresso-in","espresso-out"] :
        if args.input_unit is not None and args.input_unit != "angstrom":
            print("\t{:s}: if 'input_format' == 'espresso-io/out' only 'input_unit' == 'angstrom' (or None) is allowed. ".format(error))
            return 
        if args.input_unit_cell is not None and args.input_unit_cell != "angstrom":
            print("\t{:s}: if 'input_format' == 'espresso-io/out' only 'input_unit_cell' == 'angstrom' (or None) is allowed. ".format(error))
            return 
        
        args.input_unit = "angstrom"
        args.input_unit_cell = "angstrom"

        if args.output_unit is None:
            print("\n\t{:s}: the input file format is '{:s}', then the position ".format(warning,args.input_format)+\
                "and cell are automatically convert to 'angstrom' by ASE.\n\t"+\
                    "Specify the output units (-ou,--output_unit) if you do not want the output to be in 'angstrom'.")
        if args.output_format is None or args.output_format == "espresso-in":
            print("\n\t{:s}: the output file format is 'espresso-in'.\n\tThen, even though the positions have been converted to another unit, ".format(warning) + \
                    "you will find the keyword 'angstrom' in the output file."+\
                    "\n\t{:s}\n".format(keywords))

    #------------------#
    pbc = np.any( [ np.all(atoms[n].get_pbc()) for n in range(len(atoms)) ] )
    print("\tThe atomic structure is {:s}periodic.".format("" if pbc else "not "))

    if args.pbc is not None:
        if args.pbc and not pbc:
            raise ValueError("You required the structures to be periodic, but they are not.")
        elif not args.pbc and pbc:
            print("\tYou required to remove periodic boundary conditions.")
            print("\tRemoving cells from all the structures ... ",end="")
            for n in range(len(atoms)):
                atoms[n].set_cell(None)
                atoms[n].set_pbc(False)
            print("done")
            pbc = False

    #------------------#
    if args.output_unit is not None :
        if args.input_unit is None :
            args.input_unit = "atomic_unit"
        extra = "" if not pbc else "(and lattice parameters) "
        
        factor_pos = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit)
        if pbc : 
            if args.input_unit_cell is None :
                print("\t{:s} The unit of the lattice parameters is not specified ('input_unit_cell'):".format(warning)+\
                      "\n\t\tit will be assumed to be equal to the positions unit")
                args.input_unit_cell = args.input_unit
            # print("\tConverting lattice parameters from '{:s}' to '{:s}'".format(args.input_unit_cell,args.output_unit))
            factor_cell = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit_cell)

        print("\tConverting positions {:s}from '{:s}' to '{:s}' ... ".format(extra,args.input_unit,args.output_unit),end="")
        for n in range(len(atoms)):
            atoms[n].set_calculator(None)
            atoms[n].positions *= factor_pos
            if np.all(atoms[n].get_pbc()):
                atoms[n].cell *= factor_cell
        print("done")

    #------------------#
    # atoms[0].positions - (atoms[0].cell.array @ atoms[0].get_scaled_positions().T ).T 
    if args.rotate:
        print("\tRotating the lattice vectors such that they will be in upper triangular form ... ",end="")
        for n in range(len(atoms)):
            atom = atoms[n]
            # frac = atom.get_scaled_positions()
            cellpar = atom.cell.cellpar()
            cell = Cell.fromcellpar(cellpar).array

            atoms[n].set_cell(cell,scale_atoms=True)
        print("done")

    #------------------#
    # scale
    if args.scaled:
        print("\tReplacing the cartesian positions with the fractional/scaled positions: ... ",end="")        
        for n in range(len(atoms)):
            atoms[n].set_positions(atoms[n].get_scaled_positions())
        print("done")
        print("\n\t{:s}: in the output file the positions will be indicated as 'cartesian'.".format(warning) + \
              "\n\t{:s}".format(keywords))

    #------------------#
    # Write the data to the specified output file with the specified format
    if args.folder is None:
        print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
        try:
            write(images=atoms,filename=args.output, format=args.output_format) # fmt)
            print("done")
        except Exception as e:
            print("\n\tError: {:s}".format(e))
    else:
        print("\n\tWriting each atomic structure to a different file in folder '{:s}' ... ".format(args.folder), end="")
        if not os.path.exists(args.folder):
            os.mkdir(args.folder)
        for n,structure in enumerate(atoms):
            file_name, file_extension  = os.path.splitext(args.output)
            file = f"{args.folder}/{file_name}.n={n}{file_extension}"
            file = os.path.normpath(file)
            try:
                write(images=structure,filename=file, format=args.output_format) # fmt)
            except Exception as e:
                print("\n\tError: {:s}".format(e))
        print("done")

if __name__ == "__main__":
    main()
