#!/usr/bin/env python
import argparse
import os
import numpy as np
from copy import copy
from ase.io import write, read
from eslib.classes.properties import properties as Properties
from eslib.functions import suppress_output, get_one_file_in_folder, str2bool
from eslib.input import size_type
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt, error, warning

#---------------------------------------#
# example:
# python ipi2extxyz.py -p i-pi -f data -aa forces,data/i-pi.forces_0.xyz -ap dipole,potential -o test.extxyz

# To Do:
# - add a long description with some example how to use the script

#---------------------------------------#
# Description of the script's purpose
description = "Convert the i-PI output files to an extxyz file with the specified properties and arrays."

#---------------------------------------#
def prepare_args(description):

    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar" : "\b",}

    parser.add_argument("-p", "--prefix", type=str, default='i-pi', **argv,
                        help="prefix of the i-PI output files (default: 'i-pi')")
    
    parser.add_argument("-f", "--folder", type=str, default='.', **argv,
                        help="folder (default: '.')")
    
    parser.add_argument("-qf", "--positions_file",  type=str, default=None, **argv,
                        help="input file containing the MD trajectory positions and cells (default: '[prefix].positions_0.xyz')")
    
    parser.add_argument("-pbc", "--pbc",  type=str2bool, default=True, **argv,
                        help="whether the system is periodic (default: True)")

    parser.add_argument("-pf", "--properties_file",  type=str, default=None, **argv,
                        help="input file containing the MD trajectory properties (default: '[prefix].properties.out')")

    parser.add_argument("-if", "--format",  type=str, default='i-pi', **argv,
                        help="input file format (default: 'i-pi')")

    parser.add_argument("-aa", "--additional_arrays",  type=lambda s: size_type(s,dtype=str), default=None, **argv,
                        help="additional arrays to be added to the output file (example: [velocities,forces], default: [])")
    
    parser.add_argument("-ap", "--additional_properties",  type=lambda s: size_type(s,dtype=str), default=["all"], **argv,
                        help="additional properties to be added to the output file (example: [potential,dipole], default: [all])")

    parser.add_argument("-o", "--output",  type=str, default='output.extxyz', **argv,
                        help="output file in extxyz format (default: 'output.extxyz')")

    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    if args.positions_file is None:
        if args.prefix is None or args.prefix == "":
            raise ValueError("Please provide a prefix for the i-PI output files (--prefix) or a file with the atomic structures (--positions).")
        else :
            try :
                args.positions_file = get_one_file_in_folder(folder=args.folder,ext="xyz",pattern="positions")
            except:
                raise ValueError("Problem deducing the atomic structures file from the i-PI prefix.\n\
                                Please check that the folder (-f,--folder) and the prefix (-i,--prefix) are correct.\n\
                                Otherwise we can also directly specify the atomic structures file (-q,--positions).")
    elif not os.path.exists(args.positions_file):
        raise ValueError("File '{:s}' does not exist.".format(args.positions_file))

    print("\tReading atomic structures from file '{:s}' ... ".format(args.positions_file), end="")
    with suppress_output():
        atoms = AtomicStructures.from_file(file=args.positions_file,format=args.format)
    print("done")
    # else :
    #     raise ValueError("to be implemented yet")
    #     print("\tReading atomic structures from file '{:s}' using the 'ase.io.read' ... ".format(args.positions_file), end="")
    #     atoms = read(args.positions_file,format=args.format,index=":")
    #     print("done")

    if not args.pbc:
        print("\n\tRemoving lattice vectors from all the atomic structures ... ", end="")
        for n in range(len(atoms)):
            atoms[n].set_pbc([False, False, False])
            atoms[n].set_cell(None)
        print("done")

    if args.additional_arrays is not None:
        arrays = dict()
        for k in args.additional_arrays:
            try :
                file = get_one_file_in_folder(folder=args.folder,ext="xyz",pattern=k)
            except:
                raise ValueError("No file provided or found for array '{:s}'".format(k))
            print("\tReading additional array '{:s}' from file '{:s}' ... ".format(k,file), end="")
            tmp = read(file,index=":")
            arrays[k] = np.zeros((len(tmp)),dtype=object)
            for n in range(len(tmp)):
                arrays[k][n] = tmp[n].positions
            print("done")
        
        # atoms.arrays = dict()
        for k in arrays.keys():
            for n in range(len(arrays[k])):
                atoms[n].arrays[k] = arrays[k][n]

    if args.additional_properties is not None or "none" not in args.additional_properties:
        properties = list(args.additional_properties)
        print("\n\tYou specified the following properties to be added to the output file: ",properties)

        ###
        # properties
        if args.properties_file is None:
            if args.prefix is None or args.prefix == "":
                raise ValueError("Please provide a prefix for the i-PI output files (--prefix) or the i-PI file with the properties (--properties).")
            else :
                try :
                    args.properties_file = get_one_file_in_folder(folder=args.folder,ext="out",pattern="properties")
                except:
                    raise ValueError("Problem deducing the properties file from the i-PI prefix.\n\
                                    Please check that the folder (-d,--folder) and the prefix (-i,--prefix) are correct.\n\
                                    Otherwise we can also directly specify the properties file (-p,--properties).")
        elif not os.path.exists(args.properties_file):
            raise ValueError("File '{:s}' does not exist.".format(args.properties_file))
                
        print("\tReading properties from file '{:s}' ... ".format(args.properties_file), end="")
        with suppress_output():
            if str(args.properties_file).endswith(".pickle"):
                allproperties = Properties.from_pickle(file_path=args.properties_file)
            else:
                allproperties = Properties.load(file=args.properties_file)
        print("done")

        print("\n\tSummary:")
        print("\t# atomic structures: {:d}".format(len(atoms)))
        print("\t       # properties: {:d}".format(len(allproperties)))

        if len(allproperties) != len(atoms):
            print("\n\t{:s}: n. of atomic structures and n. of properties differ.".format(warning))
            if len(allproperties) == len(atoms)+1 :
                information = "You should provide the positions as printed by i-PI."
                try:
                    from colorama import Fore, Style
                    information = Fore.YELLOW   + Style.NORMAL + information + Style.RESET_ALL
                except:
                    pass
                print("\t{:s}\n\tMaybe you provided a 'replay' input file --> discarding the first properties raw.".format(information))
                allproperties = allproperties[1:]
            else:
                raise ValueError("I would expect n. of atomic structures to be (n. of properties + 1)")

        print("\n\tSummary of the read properties:\n")
        df = allproperties.summary()

        def line(): print("\t\t-----------------------------------------")
        
        line()
        print("\t\t|{:^15s}|{:^15s}|{:^7s}|".format("name","unit","shape"))
        line()
        for index, row in df.iterrows():
            print("\t\t|{:^15s}|{:^15s}|{:^7d}|".format(row["name"],row["unit"],row["shape"]))
        line()

        # all properties
        if "all" in properties:
            _properties = list(allproperties.properties.keys())
            for p in properties:
                p = p.replace(" ","")
                if p[0] == "~":
                    _properties.remove(p[1:])
            properties = copy(_properties)  
            del _properties
        
        print("\n\tStoring the following properties to file: ",properties)

        # 
        tmp = dict()
        for k in properties:
            tmp[k] = allproperties.properties[k]
        properties = copy(tmp)
        del allproperties
        del tmp
    
        for k in properties.keys(): 
            for n in range(len(properties[k])):
                atoms[n].info[k] = properties[k][n]

    ###
    # writing
    print("\n\tWriting output to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, list(atoms), format="extxyz") # fmt)
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

if __name__ == "__main__":
    main()
