#!/usr/bin/env python
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Summary of an MD trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input",         **argv,type=str, help="input file")
    parser.add_argument("-if", "--input_format" , **argv,type=str, help="input file format (default: 'None')" , default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done\n")

    print("\tn. of atomic structures: {:d}".format(len(atoms)))

    #---------------------------------------#
    pbc = atoms.call(lambda e:e.pbc)
    if np.all(pbc):
        print("\tperiodic (all axis): true")
    elif np.all(~pbc):
        print("\tperiodic (any axis): false")
    else:
        print("\tperiodic along axis x,y,z: ",[ str(a) for a in np.all(pbc,axis=0) == ~ np.all(~pbc,axis=0) ])

    #---------------------------------------#
    keys = atoms[0].info.keys()
    check = dict()
    
    for k in keys:
        for n in range(len(atoms)):
            if k not in atoms[n].info.keys():
                check[k] = False
                break
        check[k] = True

    print("\n\tInfo/properties shapes:")
    line = "\t\t"+"-"*21
    print(line)
    for k in keys:
        print("\t\t|{:^12s}|{:^6s}|".format(k,str(atoms[0].info[k].shape)),end="")
        if not check[k]:
            print(" not present in all the structures")
        else:
            print()
    print(line)

    #---------------------------------------#
    keys = atoms[0].arrays.keys()
    check = dict()
    
    for k in keys:
        for n in range(len(atoms)):
            if k not in atoms[n].info.keys():
                check[k] = False
                break
        check[k] = True

    print("\n\tArrays shapes:")
    line = "\t\t"+"-"*27
    print(line)
    for k in keys:
        print("\t\t|{:^12s}|{:^12s}|".format(k,str(atoms[0].arrays[k].shape)),end="")
        if not check[k]:
            print(" not present in all the structures")
        else:
            print()
    print(line)   
        
#---------------------------------------#
if __name__ == "__main__":
    main()
