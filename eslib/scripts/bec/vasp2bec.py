#!/usr/bin/env python
import re
import numpy as np
from eslib.formatting import esfmt, float_format

#---------------------------------------#
# Description of the script's purpose
description = "Read the Born Charges from a VASP calculation."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input" , **argv, required=True , type=str  , help="OUTCAR file of the VASP calculation (default: %(default)s)")
    parser.add_argument("-o" , "--output", **argv, required=False, type=str  , help="output file with the BEC tensors (default: %(default)s)", default='bec.txt')
    return parser

#---------------------------------------#
def parse_born_charges_from_file(filename):
    born_tensors = []

    start_pattern = re.compile(r"BORN EFFECTIVE CHARGES")
    ion_pattern = re.compile(r"^\s*ion\s+\d+")

    in_block = False

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:  # EOF
                break

            if not in_block:
                if start_pattern.search(line):
                    in_block = True
                    line = f.readline()
                    line = f.readline()
                else:
                    continue  # keep looking for start

            # Now inside the block
            if ion_pattern.match(line):
                # Read next 3 lines for the 3x3 matrix
                current_tensor = []
                for _ in range(3):
                    matrix_line = f.readline()
                    if not matrix_line:
                        break  # EOF reached prematurely
                    parts = matrix_line.strip().split()
                    # parts[0] is line index, next 3 are floats
                    row = list(map(float, parts[1:4]))
                    current_tensor.append(row)

                assert len(current_tensor) == 3, "error"
                born_tensors.append(current_tensor)
            else:
                # Non-matching line means block ended
                break

    return np.array(born_tensors)
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print(f"\tReading the Born Charges from file '{args.input}' ... ", end="")
    born = parse_born_charges_from_file(args.input)
    print("done")
    print("\tborn.shape: ",born.shape)
    
    born = born.reshape((born.shape[0],-1))

    #------------------#
    header = (
        "dP_x/dR_x, dP_x/dR_y, dP_x/dR_z, "
        "dP_y/dR_x, dP_y/dR_y, dP_y/dR_z, "
        "dP_z/dR_x, dP_z/dR_y, dP_z/dR_z"
    )

    # Save to the specified output file
    print(f"\tSaving Born Charges to file '{args.output}' ... ",end="")
    np.savetxt(args.output, born, fmt=float_format, header=header)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()