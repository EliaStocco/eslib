#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import argparse
import numpy as np
from elia.formatting import matrix2str
from elia.tools import find_transformation
from elia.input import str2bool
from elia.tools import convert
from ase.cell import Cell
from ase import Atoms
from ase.io import write
from gims.structure import Structure, read


#---------------------------------------#
description     = "Show general information of a given atomic structure and find its primitive cell structure using GIMS."
warning         = "***Warning***"
closure         = "Job done :)"
keywords        = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"
divisor         = "-"*100

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN  + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
    divisor         = Fore.CYAN   + Style.NORMAL + divisor         + Style.RESET_ALL
except:
    pass
    
#---------------------------------------#
def print_info(structure:Atoms,threshold:float,title:str):
    from gims.structure_info import StructureInfo
    strinfo = StructureInfo(structure,threshold)
    info = str(strinfo)
    # print("done")    

    print("\t{:s}:\n\tthe positions are assumed to be stored in angstrom in the input file".format(warning))
    info = info.replace("System Info","\n{:s}".format(title))
    info = info.replace("\n","\n\t")
    info = info.replace("Lattice parameters <br> ","")
    info = info.replace("(a, b, c, α, β, γ)","{:<30}".format("(a, b, c, α, β, γ) [ang,deg]"))
    print("\t"+info)

    print("\tAdditional information:")
    try:
        V = structure.get_volume()
        print("\t{:<30}:".format("volume [ang^3]"),V)
        factor = convert(1,"length","angstrom","atomic_unit")**3
        print("\t{:<30}:".format("volume [au^3]"),V*factor)
    except:
        pass
    print("\t{:<30}:".format("chemical symbols"),structure.get_chemical_symbols())

    # from icecream import ic
    # # ic(structure.get_cell().cellpar)
    # try:
    #     ic(strinfo.equivalent_atoms)
    # except:
    #     pass

    if np.all(structure.get_pbc()):
        print("\n\tCell:")
        line = matrix2str(structure.cell.array.T,col_names=["1","2","3"],cols_align="^",width=6)
        print(line)

        print("\n\tPositions (cartesian and fractional):")
        cartesian = structure.get_positions()
        fractional = ( np.linalg.inv(structure.get_cell().T) @ cartesian.T ).T
        M = np.concatenate([cartesian,fractional], axis=1)
        line = matrix2str(M,digits=3,col_names=["Rx","Ry","Rz","fx","fy","fy"],cols_align="^",width=8,row_names=structure.get_chemical_symbols())
        print(line)
    else:
        print("\n\tPositions (cartesian):")
        line = matrix2str(structure.get_positions(),digits=3,col_names=["Rx","Ry","Rz"],cols_align="^",width=8,row_names=structure.get_chemical_symbols())
        print(line)
    return
#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-i", "--input",  type=str,**argv,
        help="atomic structure input file"
    )
    parser.add_argument(
        "-t", "--threshold",  type=float,**argv,
        help="threshold for GIMS (default: 1e-3)", default=1e-3
    )
    parser.add_argument(
        "-r"  , "--rotate" , type=str2bool, **argv,
        help="whether to rotate the cell to the upper triangular form compatible with i-PI (default: True)", default=True
    )
    parser.add_argument(
        "-p"  , "--primitive" , type=str2bool, **argv,
        help="whether to compute the primitive structure (default: False)", default=False
    )
    parser.add_argument(
        "-o", "--output",  type=str,**argv,
        help="output file of the primitive structure (default: 'None')", default=None
    )
    parser.add_argument(
        "-of" , "--output_format",   type=str, **argv,
        help="output file format (default: 'None')", default=None
    )
    return parser.parse_args()

#---------------------------------------#
def main():

    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))

    #---------------------------------------#
    print("\n\t{:s}".format(divisor))
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    if args.rotate:
        print("\tRotating the lattice vectors of the atomic structure such that they will be in upper triangular form ... ",end="")
        # frac = atom.get_scaled_positions()
        cellpar = atoms.cell.cellpar()
        cell = Cell.fromcellpar(cellpar).array
        if np.allclose(cell,atoms.cell):
            print("done")
            print("\tThe lattice vectors are already in upper triangular form.")
        else:
            atoms.set_cell(cell,scale_atoms=True)
            print("done")

    print("\n\tComputing general information of the atomic structure using GIMS ... ",end="")
    structure = Structure(atoms)
    print("done") 
    print_info(structure,args.threshold,"Original structure information:")

    if args.primitive:
        #---------------------------------------#
        print("\n\t{:s}".format(divisor))
        print("\n\tComputing the primitive cell using GIMS ... ",end="")
        primive_structure = structure.get_primitive_cell(args.threshold)
        print("done")

        if args.rotate:
            print("\tRotating the lattice vectors of the primitive structure such that they will be in upper triangular form ... ",end="")
            # frac = atom.get_scaled_positions()
            cellpar = primive_structure.cell.cellpar()
            cell = Cell.fromcellpar(cellpar).array
            if np.allclose(cell,primive_structure.cell):
                print("done")
                print("\tThe lattice vectors are already in upper triangular form.")
            else:
                primive_structure.set_cell(cell,scale_atoms=True)
                print("done")    

        print("\n\tComputing general information of the primitive structure using GIMS ... ",end="")
        print("done") 
        print_info(primive_structure,args.threshold,"Primitive cell structure information:")
        
        #---------------------------------------#
        # trasformation
        print("\n\t{:s}".format(divisor))
        size, M = find_transformation(primive_structure,structure)
        print("\tTrasformation matrix from primitive to original cell:")
        line = matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6)
        print(line)

        #---------------------------------------#
        # Write the data to the specified output file with the specified format
        if args.output is not None:            
            print("\n\t{:s}".format(divisor))
            print("\n\tWriting primitive structure to file '{:s}' ... ".format(args.output), end="")
            try:
                write(images=primive_structure,filename=args.output, \
                      format=args.output_format) # fmt)
                print("done")
            except Exception as e:
                print("\n\tError: {:s}".format(e))

    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()