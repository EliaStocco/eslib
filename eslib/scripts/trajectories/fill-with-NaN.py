#!/usr/bin/env python
import json
import numpy as np
from ase.stress import full_3x3_to_voigt_6_stress
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Fill missing properties with NaN (it needs the JSON output of 'properties2json.py')."

ase_like_properties = {
    "energy": (),
    "interaction_energy": (),
    "node_energy": ("natoms",),
    "forces": ("natoms", 3),
    "displacement": (3, 3),
    "stress": (6,),
    "virials": (6, ),
    "dipole": (3,),
    "atomic_dipoles": ("natoms", 3),
    "atomic-oxn-dipole": ("natoms", 3),
    "BEC": ("natoms", 9),  # ("natoms", 3, 3) is not supported by ASE
    "piezoelectric": (3, 3, 3),
}

keys = list(ase_like_properties.keys())
for k in keys:
    ase_like_properties[f"REF_{k}"] = ase_like_properties[k]

#---------------------------------------#
def convert_value(v):
    """
    Convert JSON-loaded value:

    - list of int or "natoms" -> tuple
    - strings "int", "float", "str", "bool" -> corresponding Python type
    - otherwise return as-is
    """
    # Shape case: list of int or "natoms"
    if isinstance(v, list) and all(isinstance(x, int) or x == "natoms" for x in v):
        return tuple(v)

    # Dtype case: string -> Python type
    if isinstance(v, str):
        builtin_types = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        return builtin_types.get(v, v)

    return v

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str, required=True , help="input file")
    parser.add_argument("-if", "--input_format" , **argv, type=str, required=False, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-p" , "--properties"   , **argv, type=str, required=True , help="JSON input file with properties")
    parser.add_argument("-f" , "--fill_json"    , **argv, type=str, required=False, help="JSON input file with shapes (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print(f"\tn. of atomic structures: {len(structures)}")
    
    #------------------#
    if args.fill_json is not None:
        print(f"\tReading shapes from file '{args.fill_json}' ... ", end="")
        with open(args.fill_json, "r") as f:
            shapes:dict = json.load(f)
        print("done")
        for k,v in shapes.items():
            shapes[k] = convert_value(v)
    else:
        shapes = ase_like_properties
            
    #------------------#
    print("\n\tShapes summary:")
    for k,v in shapes.items():
        print(f"\t - {k}: ",v)
        
    #------------------#
    print(f"\n\tReading properties from file '{args.properties}' ... ", end="")
    with open(args.properties, "r") as f:
        properties:dict = json.load(f)
    print("done")
    print("\tProperties summary:")
    all_infos = list(properties['info'].keys())
    all_arrays = list(properties['arrays'].keys())
    print("\t -  infos: ", all_infos)
    print("\t - arrays: ", all_arrays)
    # for k,v in properties.items():
    #     print(f"\t - {k}: ",v)
    
        
    #------------------#
    print("\n\tFilling missing properties with NaN ... ",end="")
    for atoms in structures:
        Natoms = atoms.get_global_number_of_atoms()
        for info in all_infos:
            if info not in atoms.info:
                shape = shapes[info]
                v = np.full(shape,np.nan)
                atoms.info[info] = v
            elif "stress" in info or "virials" in info:
                v = atoms.info[info]
                if v.shape == (3,3):
                    atoms.info[info] = full_3x3_to_voigt_6_stress(v)
        del info
        for array in all_arrays:
            if array not in atoms.arrays:
                shape = shapes[array]
                if shape[0] == "natoms":
                    shape = tuple(Natoms,*shape[1:])
                v = np.full(shape,np.nan)
                atoms.arrays[array] = v
    print("done")
                
    #---------------------------------------#
    # summary
    print("\n\tSummary of the atomic structures: ")
    df = structures.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    
    #---------------------------------------#
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")

    return

#---------------------------------------#
if __name__ == "__main__":
    main()
