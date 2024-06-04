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
    parser.add_argument("-i", "--input", **argv, type=str, help="input file")
    parser.add_argument("-if", "--input_format", **argv, type=str, help="input file format (default: %(default)s)", default=None)
    return parser  # .parse_args()

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done\n")
    print("\tn. of atomic structures: {:d}".format(len(atoms)))

    #---------------------------------------#
    Natoms = [a.get_global_number_of_atoms() for a in atoms]
    if len(set(Natoms)) == 1:
        print(f"\n\tAll structures have the same number of atoms: {Natoms[0]}")
    else:
        print("\n\tStructures have varying numbers of atoms.")

    #---------------------------------------#
    pbc = np.asarray([np.all(a.get_pbc()) for a in atoms])
    if np.all(pbc):
        print("\tperiodic (all axis): true")
    elif np.all(~pbc):
        print("\tperiodic (any axis): false")
    else:
        print("\tperiodic along axis x,y,z: ", [str(a) for a in np.all(pbc, axis=0) == ~ np.all(~pbc, axis=0)])

    #---------------------------------------#
    keys = atoms[0].info.keys()
    check = dict()

    if len(keys) > 0:
        max_key_length = max(len(k) for k in keys)
        max_shape_length = max(len(str(atoms[0].info[k].shape)) for k in keys)

        for k in keys:
            for n in range(len(atoms)):
                if k not in atoms[n].info.keys():
                    check[k] = False
                    break
            else:
                check[k] = True

        print("\n\tInfo/properties shapes:")
        line = "\t\t" + "-" * (max_key_length + max_shape_length + 7)
        print(line)
        for k in keys:
            try:
                shape = str(atoms[0].info[k].shape)
            except:
                shape = str(type(atoms[0].info[k]))
            print("\t\t|{:^{key_width}s}|{:^{shape_width}s}|".format(k, shape, key_width=max_key_length, shape_width=max_shape_length), end="")
            if not check[k]:
                print(" not present in all the structures")
            else:
                print()
        print(line)
    else:
        print("\n\tNo info/properties found")

    #---------------------------------------#
    keys = atoms[0].arrays.keys()
    if len(keys) > 0:
        check = dict()
        max_key_length = max(len(k) for k in keys)
        max_shape_length = max(len(str(atoms[0].arrays[k].shape)) for k in keys)+2

        for k in keys:
            for n in range(len(atoms)):
                if k not in atoms[n].arrays.keys():
                    check[k] = False
                    break
            else:
                check[k] = True

        print("\n\tArrays shapes:")
        line = "\t\t" + "-" * (max_key_length + max_shape_length + 3)
        print(line)
        for k in keys:
            shape = str(atoms[0].arrays[k].shape)
            print("\t\t|{:^{key_width}s}|{:^{shape_width}s}|".format(k, shape, key_width=max_key_length, shape_width=max_shape_length), end="")
            if not check[k]:
                print(" not present in all the structures")
            else:
                print()
        print(line)
    else:
        print("\n\tNo arrays found")

#---------------------------------------#
if __name__ == "__main__":
    main()
