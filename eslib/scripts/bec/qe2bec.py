#!/usr/bin/env python
import re
import numpy as np
from ase import Atoms
from eslib.formatting import esfmt, float_format
from eslib.classes.atomic_structures import AtomicStructures

#---------------------------------------#
description = "Read Born Effective Charges and dielectric tensor from a Quantum ESPRESSO (ph.x) output file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, type=str, help="input file")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format", default=None)
    parser.add_argument("-iph", "--input_ph", **argv, required=True, type=str, help="Quantum ESPRESSO (ph.x) output file")
    parser.add_argument("-p", "--prefix", **argv, required=False, type=str, help="prefix for properties", default="REF_")
    parser.add_argument("-z" , "--bec"          , **argv, required=False, type=str, help="output file with the BEC tensors (default: %(default)s)", default='bec.txt')
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="extxyz output file", default='qe.extxyz')
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format", default=None)
    return parser

#---------------------------------------#
def parse_dielectric(file_path):
    """Parse dielectric tensor from QE output"""
    pattern = re.compile(r"Dielectric constant in cartesian axis")
    matrix_line = re.compile(r"\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)")
    epsilon = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if pattern.search(line):
            for j in range(3):
                m = matrix_line.search(lines[i + 2 + j])
                epsilon.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            break
    return np.array(epsilon)

#---------------------------------------#
def parse_born_charges(file_path, asr=True):
    """
    Parse Born effective charges from QE output
    asr=True  -> with ASR applied
    asr=False -> without ASR
    """
    if asr:
        header_pattern = re.compile(r"Effective charges .*with asr applied")
    else:
        header_pattern = re.compile(r"Effective charges .*without acoustic sum rule applied")

    matrix_line = re.compile(r"\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)")
    atom_line = re.compile(r"^\s*atom\s+\d+")

    born_tensors = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if header_pattern.search(line):
            i += 1
            while i < len(lines):
                if atom_line.search(lines[i]):
                    # skip the mean Z* line
                    i += 1
                    tensor = []
                    for _ in range(3):
                        m = matrix_line.search(lines[i])
                        tensor.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                        i += 1
                    born_tensors.append(np.array(tensor))
                else:
                    break
            break
        i += 1

    return np.array(born_tensors)

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    print(f"\tReading dielectric constant from '{args.input_ph}' ... ", end="")
    epsilon = parse_dielectric(args.input_ph)
    print("done")
    print("\tDielectric constant:")
    print("\t   x y z")
    print("\t x ",*list(epsilon[:,0]))
    print("\t y ",*list(epsilon[:,1]))
    print("\t z ",*list(epsilon[:,2]))

    print(f"\tReading Born charges (with ASR) from '{args.input_ph}' ... ", end="")
    born_asr = parse_born_charges(args.input_ph, asr=True)
    print("done")

    # reshape to (natoms, 9)
    born = born_asr.reshape((born_asr.shape[0], -1))

    # Save Born charges
    header = (
        "dP_x/dR_x, dP_x/dR_y, dP_x/dR_z, "
        "dP_y/dR_x, dP_y/dR_y, dP_y/dR_z, "
        "dP_z/dR_x, dP_z/dR_y, dP_z/dR_z"
    )
    print(f"\tSaving Born charges to '{args.bec}' ... ", end="")
    np.savetxt(args.bec, born, fmt=float_format, header=header)
    print("done")

    # Read structure from QE output using ASE (assuming the file is extended XYZ compatible)
    print(f"\tReading structure from '{args.input}' with ASE ... ", end="")
    atoms: Atoms = AtomicStructures.from_file(file=args.input, format=args.input_format,index=0)[0]  # may need to adapt format
    print("done")

    # attach BECs and dielectric tensor
    atoms.arrays[f"{args.prefix}BEC"] = born
    atoms.arrays[f"{args.prefix}BECx"] = born[:, 0:3]
    atoms.arrays[f"{args.prefix}BECy"] = born[:, 3:6]
    atoms.arrays[f"{args.prefix}BECz"] = born[:, 6:9]

    atoms.info[f"{args.prefix}epsilon"] = epsilon
    atoms.info[f"{args.prefix}epsilon_sym"] = 0.5 * (epsilon + epsilon.T)

    structures = AtomicStructures([atoms])

    # Write extended XYZ with BECs included
    print(f"\n\tWriting atomic structure to '{args.output}' ... ", end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
