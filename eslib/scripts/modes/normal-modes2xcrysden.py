#!/usr/bin/env python
import os
import numpy as np
from ase import Atoms
from typing import List
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt, warning
from eslib.tools import convert, read_file_content
from eslib.classes.trajectory import AtomicStructures

#---------------------------------------#
# Description of the script's purpose
description = "Normal Modes to xcrysden"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input" , **argv, required=True, type=str, help="*.pickle normal modes file")
    parser.add_argument("-o", "--output", **argv, required=True, type=str, help="output folder (default: %(default)s)", default='xsf-modes')
    return parser

#---------------------------------------#
def get_xsf_content(atoms: Atoms, array_name: str = None) -> str:
    """
    Generates an XCrySDen-compatible .xsf content as a string from an ASE Atoms object, 
    with positions on the left and optional array data on the right.
    
    Parameters:
        atoms (ase.Atoms): The input ASE Atoms object.
        array_name (str): The name of the array in the Atoms object to include (e.g., forces).
                         If None, no additional array data will be included.
    
    Returns:
        str: The .xsf content as a string.
    """
    # Extract unit cell
    cell = atoms.cell.array
    
    # Extract atomic numbers and positions
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    
    # Extract the additional array if provided
    array_data = None
    if array_name:
        if array_name in atoms.arrays:
            array_data = atoms.arrays[array_name]
        else:
            raise ValueError(f"Array '{array_name}' not found in the Atoms object.")
    
    # Generate the content
    content = []
    content.append("CRYSTAL")
    content.append("PRIMVEC")
    for vec in cell:
        content.append(f" {vec[0]:18.14f} {vec[1]:18.14f} {vec[2]:18.14f}")
    
    content.append("PRIMCOORD")
    content.append(f" {len(atoms)} 1")
    
    # Write atomic positions and additional array data if available
    for i, (number, pos) in enumerate(zip(atomic_numbers, positions)):
        line = f" {number:2d} " + " ".join(f"{p:18.14f}" for p in pos)
        if array_data is not None:
            array_values = array_data[i]
            line += " " + " ".join(f"{val:18.14f}" for val in array_values)
        content.append(line)
    
    # Join the lines with newlines
    return "\n".join(content)

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading phonon modes from file '{:s}' ... ".format(args.input), end="")
    nm = NormalModes.from_pickle(args.input)
    print("done")

    #---------------------------------------#
    modes = nm.get("mode").real
    if np.any(np.isnan(modes)):
        print(f"\t{warning}: Normal Modes are NaN. Let's try to compute them right now from eigvec.")
        nm.eigvec2modes()
        modes = nm.get("mode").real
        
    #---------------------------------------#   
    atoms = nm.reference
    factor = convert(1,"length","atomic_unit","angstrom")
    atoms.positions *= factor
    atoms.set_cell(np.asarray(atoms.cell) * factor)
    structures = AtomicStructures.from_atoms(atoms,repeat=len(modes.T),clean=True)

    #---------------------------------------#
    # frequencies
    print("\tComputing frequencies ... ", end="")
    eigvals = nm.get("eigval")
    freqs = np.asarray([np.sqrt(a) if a > 0 else np.nan for a in eigvals])
    freqs = convert(freqs,"frequency","atomic_unit","thz")
    print("done")
    
    #---------------------------------------#
    
    # atoms.arrays["displacement"] = np.zeros_like(atoms.get_positions())
    for n, vec in enumerate(modes.T):
        structures[n].info["title"] = f"{freqs[n]} THz (mode {n})"
        structures[n].arrays["forces"] = np.zeros_like(structures[n].get_positions())
        structures[n].arrays["forces"][:,:] = np.asarray(vec).reshape(-1, 3)
        
    #---------------------------------------#
    print("\n\tWriting vibrational modes to folder '{:s}':".format(args.output))
    this_folder = os.path.dirname(os.path.abspath(__file__))
    start = read_file_content(f"{this_folder}/xcrysden_files/start.txt")
    end = read_file_content(f"{this_folder}/xcrysden_files/end.txt")
    os.makedirs(args.output, exist_ok=True)
    for n,struc in enumerate(structures):
        filename = os.path.join(args.output,f"mode-{n}.xsf")
        print(f"\t - mode {n} to file '{filename}' ... ", end="")
        string = get_xsf_content(struc, array_name="forces")        
        with open(filename, "w") as f:
            f.write(start)
            f.write(string)
            f.write(end)
        print("done")
        
#---------------------------------------#
if __name__ == "__main__":
    main()