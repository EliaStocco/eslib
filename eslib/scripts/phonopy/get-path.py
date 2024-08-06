#!/usr/bin/env python
from ase import Atoms
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import str2bool
import seekpath
from ase import Atoms
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
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="output file (default: %(default)s)", default="phonopy.conf")
    # parser.add_argument("-p" , "--plot"          , **argv, required=False, type=str, help="plot (default: %(default)s)", default="band.pdf")
    # parser.add_argument("-of", "--output_format" , **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

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
    seekpath_output = seekpath.get_explicit_k_path_orig_cell(primitive_structure,with_time_reversal=args.time_reversal)
    print("done")

    # Extract band path information
    band_path = seekpath_output['path']
    point_coords:dict = seekpath_output['point_coords']

    # Step 6: Print the high symmetry points and their coordinates
    print("\n\tHigh Symmetry Points:")
    for point, coord in point_coords.items():
        print("\t{:>8s}: [{:>8.3f}, {:>8.3f}, {:>8.3f}]".format(point,coord[0],coord[1],coord[2]))

    print("\n\tBand path:")
    for path in band_path:
        print("\t{:>8s} --- {:<8s}".format(path[0],path[1]))

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
    print("\tSaving the BAND and QPOINTS to file '{:s}' ... ".format(args.output), end="")
    # Step 5: Save the BAND string to the file
    with open(args.output, 'w') as f:
        f.write("BAND = " + band_string + "\n")
        f.write("\nQPOINTS = " + qpoints_string + "\n")
    print("done")
   
    # #------------------#
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title('Band Path in 3D k-space')
    # ax.set_xlabel('kx')
    # ax.set_ylabel('ky')
    # ax.set_zlabel('kz')

    # # Plot high-symmetry points
    # for k, v in point_coords.items():
    #     ax.scatter(v[0], v[1], v[2], color='red')  # Red dot for high-symmetry point
    #     ax.text(v[0], v[1], v[2], k)

    # # Plot paths between high-symmetry points
    # for segment in band_path:
    #     start_coord = point_coords[segment[0]]
    #     end_coord = point_coords[segment[1]]
        
    #     # Draw a line between the points
    #     ax.plot([start_coord[0], end_coord[0]], 
    #             [start_coord[1], end_coord[1]], 
    #             [start_coord[2], end_coord[2]], 
    #             color='blue')  # Blue line for path
        
    # plt.savefig(args.plot)
   

#---------------------------------------#
if __name__ == "__main__":
    main()