#!/usr/bin/env python
import re
import numpy as np
from ase.io import read
from eslib.formatting import esfmt, float_format
from eslib.io_tools import pattern2sorted_files
from concurrent.futures import ProcessPoolExecutor, as_completed

#---------------------------------------#
description = "Extract FHI-aims forces with and without D3 into ASE Atoms objects."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input" ,**argv, type=str, required=True , help="FHI-aims output file)")
    parser.add_argument("-o", "--output",**argv, type=str, required=False, help="prefix for the output files (default: %(default)s)", default="forces")
    return parser

#---------------------------------------#
def parse_forces_fhiaims(filename, n_atoms):
    """Parse FHI-aims output file for forces with and without D3."""
    with open(filename, "r") as f:
        text = f.read()

    # Regex: "Total forces( n ) :  x  y  z"
    pattern = re.compile(
        r"Total forces\(\s*\d+\)\s*:\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
    )
    matches = pattern.findall(text)
    forces = np.array(matches, dtype=float)

    if len(forces) < 2 * n_atoms:
        raise RuntimeError(
            f"File {filename} does not contain two full force sets "
            f"(found {len(forces)} entries, expected {2*n_atoms})."
        )

    forces_no_d3 = forces[:n_atoms]
    forces_with_d3 = forces[n_atoms:2*n_atoms]
    return forces_no_d3, forces_with_d3

#---------------------------------------#
def remove_translation(coords, forces):
    """
    Remove translational components of forces (periodic case).
    Faithful to FHI-aims implementation.

    Parameters
    ----------
    coords : (n_atoms, 3) ndarray
        Atomic coordinates (not used here, included for API consistency).
    forces : (n_atoms, 3) ndarray
        Forces acting on atoms.

    Returns
    -------
    forces_clean : (n_atoms, 3) ndarray
        Forces with translation removed.
    """
    n_atoms = coords.shape[0]
    f = forces.copy()

    # Define normalized translation vectors (independent of structure)
    inv_norm = 1.0 / np.sqrt(n_atoms)
    translation_vectors = np.zeros((3, n_atoms, 3))
    for i in range(3):
        translation_vectors[i, :, i] = inv_norm

    # Project out translation components
    for i in range(3):
        v = translation_vectors[i].ravel()
        proj = np.dot(f.ravel(), v)
        f -= proj * v.reshape(n_atoms, 3)

    return f

#---------------------------------------#
def process_file(n, file):
    """Worker function: process one FHI-aims output file."""
    atoms_ref = read(file, format="aims-output")
    n_atoms = len(atoms_ref)

    forces_no_d3, forces_with_d3 = parse_forces_fhiaims(file, n_atoms)
    forces_no_d3 = remove_translation(atoms_ref.get_positions(), forces_no_d3)
    forces_with_d3 = remove_translation(atoms_ref.get_positions(), forces_with_d3)

    assert np.allclose(forces_with_d3, atoms_ref.get_forces(), atol=1e-6), \
        f"Forces do not match those in {file}!"

    return n, forces_no_d3, forces_with_d3, file

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    files = pattern2sorted_files(args.input)
    print(f"\tFound {len(files)} files matching the pattern '{args.input}'")
    print()
    N = len(files)
    all_forces_no_d3 = [None] * N
    all_forces_with_d3 = [None] * N
    
    # Use all available cores by default
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, n, file): n for n, file in enumerate(files)}

        for future in as_completed(futures):
            n, forces_no_d3, forces_with_d3, file = future.result()
            all_forces_no_d3[n] = forces_no_d3
            all_forces_with_d3[n] = forces_with_d3
            print(f"\t{n+1}/{N}) Processing {file} ... done")


    # for n,file in enumerate(files):
    #     print(f"\t{n+1}/{N}) Processing {file} ...",end="")

    #     # Load reference structure (needed to build Atoms)
    #     atoms_ref = read(file, format="aims-output")

    #     n_atoms = len(atoms_ref)
    #     forces_no_d3, forces_with_d3 = parse_forces_fhiaims(file, n_atoms)
        
    #     forces_no_d3 = remove_translation(atoms_ref.get_positions(), forces_no_d3)
    #     forces_with_d3 = remove_translation(atoms_ref.get_positions(), forces_with_d3)
        
    #     assert np.allclose(forces_with_d3, atoms_ref.get_forces(),atol=1e-6), "Forces do not match those in the file!"
        
    #     all_forces_no_d3[n] = forces_no_d3
    #     all_forces_with_d3[n] = forces_with_d3
        
    #     print("done")
    
    forces_no_d3 = np.array(all_forces_no_d3)
    forces_with_d3 = np.array(all_forces_with_d3)
    
    print(f"\n\tForces without D3: shape {forces_no_d3.shape}")
    print(f"\tForces with D3: shape {forces_with_d3.shape}")
    
    forces_no_d3 = forces_no_d3.reshape((-1, 3))
    forces_with_d3 = forces_with_d3.reshape((-1, 3))
    
    file = f"{args.output}.no-D3.txt"
    print(f"\tWriting forces without D3 to file '{file}' ...",end="")
    np.savetxt(file, forces_no_d3,fmt=float_format)
    print("done")
    
    file = f"{args.output}.D3.txt"
    print(f"\tWriting forces with D3 to file '{file}' ...",end="")
    np.savetxt(file, forces_with_d3,fmt=float_format)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()

