#!/usr/bin/env python
import seekpath
import json
from ase import Atoms
from itertools import product

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functions import suppress_output
from eslib.input import str2bool, ilist

# # import matplotlib.pyplot as plt
# import warnings

# # Suppress all DeprecationWarnings
# warnings.simplefilter("ignore", category=DeprecationWarning)


#---------------------------------------#
# Description of the script's purpose
description = "Get a phonon band path of a crystal structure using SeeK-path."

documentation = \
    "Official documentation: https://seekpath.readthedocs.io/en/latest/index.html\n" +\
    "GitHub page: https://github.com/giovannipizzi/seekpath"


#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-tr", "--time_reversal" , **argv, required=False, type=str2bool, help="time reversal (default: %(default)s)", default=True)
    parser.add_argument("-n" , "--names"         , **argv, required=False, type=str  , help="JSON file with the names of the phonon modes (default: %(default)s)", default=None)
    # parser.add_argument("-s" , "--size"          , **argv, required=False, type=ilist  , help="size of the considered supercell (default: %(default)s)", default=None)
    parser.add_argument("-t" , "--tolerance"     , **argv, required=False, type=float,  help="spglib tolerance (default: %(default)s)" , default=1e-3)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file (default: %(default)s)", default=None)
    # parser.add_argument("-p" , "--plot"          , **argv, required=False, type=str, help="plot (default: %(default)s)", default="band.pdf")
    # parser.add_argument("-of", "--output_format" , **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

def point2LaTeX(point):
    if point in  ["GAMMA","G"]:
        return r"$\Gamma$"
    elif len(point) == 1:
        # in  ["X","T","L","W","M","F","Y","Z"]:
        return point
    elif len(point) == 3:
        return "$\\rm " + "{:s}".format(point[0]) + "_{" + "{:s}".format(point[2]) + "}$"
    else:
        raise ValueError("unknown point: '{:s}'".format(point))
    
def band_path2labels(band_path):
    labels = []
    last = None
    for path in band_path:
        if len(labels) == 0:
            labels.append(point2LaTeX(path[0]))
        elif last != path[0]:
            labels[-1] = "{}|{}".format(labels[-1],point2LaTeX(path[0]))
        labels.append(point2LaTeX(path[1]))
        # print("\t{:>8s} --- {:<8s}".format(path[0],path[1]))
        last = path[1]
    return labels
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # Step 1: Read the crystal structure
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    print("\tPreparing variables ... ", end="")
    # Step 2: Convert ASE Atoms to a format Seekpath understands (tuple of (cell, positions, atomic numbers))
    cell = structure.get_cell()
    positions = structure.get_scaled_positions()
    atomic_numbers = structure.get_atomic_numbers()

    # Seekpath input format
    primitive_structure = (cell, positions, atomic_numbers)
    print("done")

    #------------------#
    print("\tInvoking SeeK-path ... ", end="")
    # Step 3: Use Seekpath to standardize the cell and get the band path
    with suppress_output():
        # warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
        seekpath_output = seekpath.get_explicit_k_path_orig_cell(primitive_structure,with_time_reversal=args.time_reversal,symprec=args.tolerance)
    print("done")

    # Extract band path information
    band_path = seekpath_output['path']
    point_coords:dict = seekpath_output['point_coords']

    #------------------#
    # Step 6: Print the high symmetry points and their coordinates
    print("\n\tHigh Symmetry Points:")
    for point, coord in point_coords.items():
        print("\t{:>8s}: [{:>8.3f}, {:>8.3f}, {:>8.3f}]".format(point,coord[0],coord[1],coord[2]))
        
    #------------------#
    print("\n\tList of all High Symmetry Points (for phonopy qpoints.conf):")
    hsp_list = "\t"
    for point, coord in point_coords.items():
        tmp = "{:>8.3f} {:>8.3f} {:>8.3f}".format(*coord)
        hsp_list += "    {:s}    \n\t".format(tmp)
    print(hsp_list)
    
    # #------------------#
    # if args.size is not None:
    #     print("\n\tList of all commensurate points (for phonopy qpoints.conf):")
    #     def generate_tuples(divisors):
    #         x, y, z = divisors
    #         # Generate all combinations for n1, n2, n3 in the specified ranges
    #         tuples = [(n1 / x, n2 / y, n3 / z) for n1, n2, n3 in product(range(x), range(y), range(z))]
    #         return tuples
        
    #     points = generate_tuples(args.size)
    #     pass
    
    #------------------#        
    if args.names is not None:
        print("\n\tSaving the names of the high symmetry points to file '{:s}' ... ".format(args.names), end="")
        names = {}
        for point, coord in point_coords.items():
            key = "[{:.3f},{:.3f},{:.3f}]".format(coord[0],coord[1],coord[2])
            names[key] = point
        with open(args.names,'w') as f:
            json.dump(names,f,indent=4)
        print("done")

    #------------------#
    print("\n\tBand path (from seekpath):")
    for n,path in enumerate(band_path):
        print("\t{:>3d}:{:>8s} --- {:<8s}".format(n+1,path[0],path[1]))

    labels = band_path2labels(band_path)
    print("\tLabels: ",labels)

    assert len(labels) - 1 == len(band_path), "coding error"

    #------------------#
    print("\n\tBand path (to be used in phonopy):")
    labels = []
    last = None
    phonopy_band_path = []
    for n,path in enumerate(band_path):
        if n == 0 :
            phonopy_band_path.append(path)
        elif path[0] == last:
            phonopy_band_path.append(path)
        else:
            phonopy_band_path.append((last,path[0]))
            phonopy_band_path.append(path)
        last = path[1]

    for n,path in enumerate(phonopy_band_path):
        print("\t{:>3d}:{:>8s} --- {:<8s}".format(n+1,path[0],path[1]))
        
    phonopy_labels = band_path2labels(phonopy_band_path)
    print("\tLabels: ",phonopy_labels)

    assert len(phonopy_labels) - 1 == len(phonopy_band_path), "coding error"

    labels = phonopy_labels
    band_path = phonopy_band_path

    #------------------#
    print("\n\tPreparing BAND and QPOINTS ... ", end="")
    # Step 4: Format the band path as a string for the Phonopy configuration file
    band_string = ""
    for segment in band_path:
        start_point = point_coords[segment[0]]
        end_point = point_coords[segment[1]]
        band_string += " ".join(map(str, start_point)) + " "
        band_string += " ".join(map(str, end_point)) + " "

    qpoints_string = ""
    for kpoint in point_coords.values():
        qpoints_string += " ".join(map(str, kpoint)) + " "

    # Remove trailing space
    band_string = band_string.strip()
    qpoints_string = qpoints_string.strip()
    print("done")

    #------------------#
    if args.output is not None:
        print("\tSaving the BAND and QPOINTS to file '{:s}' ... ".format(args.output), end="")
        # Step 5: Save the BAND string to the file
        with open(args.output, 'w') as f:
            f.write("BAND = " + band_string + "\n")
            f.write("\nQPOINTS = " + qpoints_string + "\n")
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()