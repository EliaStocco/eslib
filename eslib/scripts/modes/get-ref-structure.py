#!/usr/bin/env python
from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt
from eslib.show import matrix2str
from eslib.tools import convert

#---------------------------------------#
THRESHOLD = 1e-4
CHECK = False
#---------------------------------------#
# Description of the script's purpose
description = "Save the reference structure of some normal modes to file."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="nomral modes file (default: %(default)s)")
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str, help="output file (default: %(default)s)", default="ref.ang.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading phonon modes from file '{:s}' ... ".format(args.input), end="")
    nm = NormalModes.from_pickle(args.input)
    print("done")
    print("\tn. of modes:",nm.Nmodes)  
    print("\tn. of dof  :",nm.Ndof)  
    
    #------------------#
    structure = nm.reference
    Natoms = structure.get_global_number_of_atoms()
    print("\tn. of atoms:",Natoms)    
    if Natoms*3 == nm.Nmodes:
        print("\tsupercell  : false --> these are vibrational modes")        
    else:
        print("\tsupercell  : true --> these are phonon modes")    
    print("\tunit cell [au] :")    
    line = matrix2str(structure.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    #------------------#
    print("\n\tConverting the reference structure from atomic units to angstrom ... ", end="")
    structure.positions *= convert(1,"length","atomic_unit","angstrom")
    structure.cell *= convert(1,"length","atomic_unit","angstrom")
    print("done")
    
    #------------------#
    print("\n\tWriting reference structure to file '{:s}' ... ".format(args.output), end="")
    structure = AtomicStructures([structure])
    structure.to_file(file=args.output,format=args.output_format)
    print("done")    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
