#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.io import write

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.structure import Structure, rotate2LT
from eslib.formatting import esfmt, warning
from eslib.input import str2bool
from eslib.show import matrix2str
from eslib.tools import convert, find_transformation

#---------------------------------------#
description     = "Show general information of a given atomic structure and find its primitive cell structure."
divisor         = "-"*100

choices = ['niggli','minkowski','conventional','primitive','phonopy-primitive']
    
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , type=str     , **argv, help="atomic structure input file")
    parser.add_argument("-if", "--input_format"  , type=str     , **argv, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-t" , "--threshold"     , type=float   , **argv, help="symmetry threshold (default: %(default)s)", default=1e-3)
    parser.add_argument("-r" , "--rotate"        , type=str2bool, **argv, help="whether to rotate the cell to the upper triangular (default: %(default)s)", default=True)
    parser.add_argument("-s" , "--shift"         , type=str2bool, **argv, help="shift the first atom to the origing (default: %(default)s)", default=False)
    parser.add_argument("-sp", "--show_positions", type=str2bool, **argv, help="show positions (default: %(default)s)", default=False)
    parser.add_argument("-c" , "--conversion"    , type=str     , **argv, help=f"structure conversion form {choices}"+" (default: %(default)s)", default=None, choices=choices)
    parser.add_argument("-o" , "--output"        , type=str     , **argv, help="output file of the converted structure (default: %(default)s)", default=None)
    parser.add_argument("-of", "--output_format" , type=str     , **argv, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
def print_info(structure:Atoms,threshold:float,title:str,show_pos:bool):
    from eslib.classes.structure import StructureInfo
    strinfo = StructureInfo(structure,threshold)
    info = str(strinfo)

    print("\t{:s}:\n\tthe positions are assumed to be stored in angstrom in the input file".format(warning))
    info = info.replace("System Info","\n{:s}".format(title))
    info = info.replace("\n","\n\t")
    info = info.replace("Lattice parameters <br> ","")
    info = info.replace("(a, b, c, α, β, γ)","{:<30}".format("(a, b, c, α, β, γ) [ang,deg]"))
    print("\t"+info)

    print("\tAdditional information:")
    # try:
    V = structure.get_volume()
    print("\t{:<30}:".format("volume [ang^3]"),V)
    factor = convert(1,"length","angstrom","atomic_unit")**3
    print("\t{:<30}:".format("volume [au^3]"),V*factor)
    tot_mass = structure.get_masses().sum()
    tot_mass = convert(tot_mass,"mass","dalton","atomic_unit")
    V = V*convert(1,"length","angstrom","atomic_unit")**3
    density = tot_mass/V
    density = convert(density,"density","atomic_unit","g/cm3")
    print("\t{:<30}:".format("density [g/cm^3]"),density)
    print("\t{:<30}:".format("chemical symbols"),structure.get_chemical_symbols())

    if np.all(structure.get_pbc()):
        print("\n\tCell:")
        line = matrix2str(structure.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
        print(line)

        if show_pos:
            print("\n\tPositions (cartesian and fractional):")
            cartesian = structure.get_positions()
            fractional = ( np.linalg.inv(structure.get_cell().T) @ cartesian.T ).T
            M = np.concatenate([cartesian,fractional], axis=1)
            line = matrix2str(M,digits=3,col_names=["Rx","Ry","Rz","fx","fy","fy"],cols_align="^",width=8,row_names=structure.get_chemical_symbols())
            print(line)
    else:
        if show_pos:
            print("\n\tPositions (cartesian):")
            line = matrix2str(structure.get_positions(),digits=3,col_names=["Rx","Ry","Rz"],cols_align="^",width=8,row_names=structure.get_chemical_symbols())
            print(line)
    return

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    converted_structure = None
    
    #---------------------------------------#
    print("\n\t{:s}".format(divisor))
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    if args.rotate:
        print("\tRotating the lattice vectors of the atomic structure such that they will be in upper triangular form ... ",end="")
        atoms = rotate2LT(atoms)
        print("done")
    
    if args.shift:
        print("\tShifting the first atom to the origin ... ",end="")
        pos = atoms.get_positions()
        v = pos[0,:]
        pos -= v
        atoms.set_positions(pos)
        print("done")


    print("\n\tComputing general information of the atomic structure ... ",end="")
    structure = Structure(atoms)
    print("done") 
    
    
    print_info(structure,args.threshold,"Original structure information:",args.show_positions)

    if args.conversion is not None:
        args.conversion  = str(args.conversion )
        #---------------------------------------#
        print("\n\t{:s}".format(divisor))
        print("\n\tConverting the structure to '{:s}' ... ".format(args.conversion),end="")
        converted_structure = structure.copy()
        if args.conversion == "niggli":
            from ase.build import niggli_reduce
            niggli_reduce(converted_structure)
        elif args.conversion == "minkowski":
            raise ValueError("not implemented yet")
            from ase.geometry.minkowski_reduction import minkowski_reduce
            minkowski_reduce(converted_structure)
        elif args.conversion == "primitive":
            converted_structure = converted_structure.get_primitive_cell(args.threshold)
        elif args.conversion == "conventional":
            converted_structure.get_conventional_cell(args.threshold)
        elif args.conversion == "phonopy-primitive":
            from phonopy.structure.cells import get_primitive, get_supercell

            from eslib.tools import ase2phonopy, phonopy2ase
            tmp = converted_structure.get_primitive_cell(args.threshold)
            tmp = tmp.rotate2LT()
            _ , M  = find_transformation(tmp,converted_structure)
            M = np.round(M,0).astype(int)
            converted_structure = ase2phonopy(converted_structure)
            primitive_matrix = np.linalg.inv(M)
            converted_structure = get_primitive(converted_structure,primitive_matrix=primitive_matrix,symprec=args.threshold)
            converted_structure = phonopy2ase(converted_structure)
        else:
            raise ValueError("coding error")
        print("done")
        
        if args.rotate:
            print("\n\tRotating the lattice vectors of the converted structure such that they will be in upper triangular form ... ",end="")
            atoms = rotate2LT(atoms)
            print("done")

        print("\n\tComputing general information of the converted structure ... ",end="")
        print("done") 
        print_info(converted_structure,args.threshold,"Converted cell structure information:",args.show_positions)

    #---------------------------------------#
    # Write the data to the specified output file with the specified format
    if args.output is not None: 
        if converted_structure is None:
            converted_structure = atoms
        print("\n\t{:s}".format(divisor))
        print("\n\tWriting converted structure to file '{:s}' ... ".format(args.output), end="")
        try:
            write(images=converted_structure,filename=args.output, format=args.output_format)
            print("done")
        except Exception as e:
            print("\n\tError: {:s}".format(e))

#---------------------------------------#
if __name__ == "__main__":
    main()