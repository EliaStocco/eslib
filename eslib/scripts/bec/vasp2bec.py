#!/usr/bin/env python
import re
import struct
import numpy as np
from ase import Atoms
from eslib.formatting import esfmt, float_format
from eslib.mathematics import voigt_to_tensor
from eslib.classes.atomic_structures import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Read the Born Effective Charges (and other quantities) from a VASP output file."
documentation = \
"The following quantities will be read and saved with the following keywords:\n\
 - Born Effective Charges: 'BECx', 'BECy', 'BECz' with shapes (natoms,3) and 'BEC' with shape (natoms,9) [same order as described in the --bec file] \n\
 - piezoelectric tensors: 'piezo_voigt' with shape (3,6) and 'piezo_full' with shape (3,3,3)\n\
 - macroscopic static dielectric tensor: 'epsilon' and 'epsilon_sym' with shape (3,3)\
"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="OUTCAR file of the VASP calculation")
    parser.add_argument("-p" , "--prefix"       , **argv, required=False, type=str, help="prefix for the keyword (default: %(default)s)", default="REF_")
    parser.add_argument("-z" , "--bec"          , **argv, required=False, type=str, help="output file with the BEC tensors (default: %(default)s)", default='bec.txt')
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str, help="extxyz output file (default: %(default)s)", default='vasp.extxyz')
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
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

def parse_epsilon_from_file(filename):
    epsilon = np.zeros((3, 3))

    start_pattern = re.compile(
        r"MACROSCOPIC STATIC DIELECTRIC TENSOR \(including local field effects in DFT\)"
    )

    with open(filename, 'r') as f:
        for line in f:
            # Look for the header line
            if start_pattern.search(line):
                # Skip the dashed line
                f.readline()

                # Read the next 3 lines containing the matrix
                matrix_lines = [f.readline() for _ in 3 * [None]]

                # Parse numbers
                for i, row in enumerate(matrix_lines):
                    epsilon[i] = np.fromstring(row, sep=' ')

                break  # Done

    return epsilon
            
def parse_piezo_from_file(filename):
    piezo = np.zeros((3, 6))

    start_pattern = re.compile(
        r"PIEZOELECTRIC TENSOR \(including local field effects\)\s+for field in x, y, z\s+\(C/m\^2\)"
    )

    with open(filename, 'r') as f:
        for line in f:
            # Look for the header line
            if start_pattern.search(line):
                # Skip the line with column labels (XX YY ZZ XY YZ ZX)
                f.readline()
                # Skip the dashed line
                f.readline()

                # Now read the next 3 lines for x, y, z
                for i in range(3):
                    row = f.readline().strip()
                    # split: first token is x/y/z label, the rest are numbers
                    parts = row.split()
                    values = list(map(float, parts[1:7]))
                    piezo[i, :] = values
                break

    return piezo

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):

    #------------------#
    # trajectory
    print(f"\tReading the Born Charges from file '{args.input}' ... ", end="")
    born = parse_born_charges_from_file(args.input)
    print("done")
    print("\tborn.shape: ",born.shape)
    
    born = born.reshape((born.shape[0],-1))
    
    header = (
        "dP_x/dR_x, dP_x/dR_y, dP_x/dR_z, "
        "dP_y/dR_x, dP_y/dR_y, dP_y/dR_z, "
        "dP_z/dR_x, dP_z/dR_y, dP_z/dR_z"
    )

    # Save to the specified output file
    print(f"\tSaving Born Charges to file '{args.bec}' ... ",end="")
    np.savetxt(args.bec, born, fmt=float_format, header=header)
    print("done")
    
    #------------------#
    print(f"\tReading the dielectric tensor tensor from file '{args.input}' ... ", end="")
    epsilon = parse_epsilon_from_file(args.input)
    print("done")
    print("\tepsilon.shape: ",epsilon.shape)
    
    #------------------#
    print(f"\tReading the piezoelectric tensor tensor from file '{args.input}' ... ", end="")
    piezo = parse_piezo_from_file(args.input)
    print("done")
    print("\tepsilon.shape: ",piezo.shape)
    
    #------------------#
    # Read the input with ASE and add the Born charges to atoms.arrays
    print(f"\tReading structure from '{args.input}' with ASE ... ", end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input, format='vasp-out',index=0)[0]
    print("done")

    # Attach Born effective charges to the atoms object
    if born.shape[0] != len(atoms):
        raise ValueError(
            f"Number of Born charge tensors ({born.shape[0]}) does not match "
            f"number of atoms ({len(atoms)})!"
        )
    
    atoms.arrays[f"{args.prefix}BEC"] = born
    atoms.arrays[f"{args.prefix}BECx"] = born[:,0:3]
    atoms.arrays[f"{args.prefix}BECy"] = born[:,3:6]
    atoms.arrays[f"{args.prefix}BECz"] = born[:,6:9]
    
    atoms.info[f"{args.prefix}epsilon"] = epsilon
    atoms.info[f"{args.prefix}epsilon_sym"] = 0.5*(epsilon+epsilon.T)
    
    atoms.info[f"{args.prefix}piezo_voigt"] = piezo
    atoms.info[f"{args.prefix}piezo_full"] = voigt_to_tensor(piezo)
    
    for key in ["magmom", "stress", "energy", "free_energy","forces"]:
        if key in atoms.info:
            atoms.info[f"{args.prefix}{key}"] = atoms.info.pop(key)
        if key in atoms.arrays:
            atoms.arrays[f"{args.prefix}{key}"] = atoms.arrays.pop(key)

    structures = AtomicStructures([atoms])
    
    #------------------#
    # summary
    try:
        print("\n\tSummary of the properties: ")
        df = structures.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t"))
    except:
        pass

    # Write extended XYZ with BECs included
    print(f"\n\tWriting atomic structure to '{args.output}' ... ", end="")    
    structures.to_file(file=args.output, format=args.output_format)
    print("done")

    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()