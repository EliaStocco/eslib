"""
Symmetrize a vibrational mode according to a target irrep (Ag or Bg).

Author: Paolo Lazzaroni
Assisted by ChatGPT
"""

import numpy as np
import spglib
from ase.io import read
from scipy.spatial.distance import cdist
import itertools
import argparse

# === Parse command line arguments ===
parser = argparse.ArgumentParser(description="Symmetrize vibrational modes by irreducible representation.")
parser.add_argument("xyz_file", type=str, help="Input XYZ file with optimized structure")
parser.add_argument("mode_file", type=str, help="File containing raw mode matrix")
parser.add_argument("mode_column", type=int, help="0-indexed column number of the mode vector")
parser.add_argument("target_irrep", type=str, choices=['Ag', 'Bg'], help="Target irreducible representation (e.g. 'Ag' or 'Bg')")
args = parser.parse_args()

# === USER OPTIONS ===
xyz_file = args.xyz_file
mode_file = args.mode_file
mode_column = args.mode_column
target_irrep = args.target_irrep
# =====================

# Step 1: Load structure using ASE
atoms = read(xyz_file)
lattice = atoms.get_cell().array
positions = atoms.get_scaled_positions()
numbers = atoms.get_atomic_numbers()

# Step 2: Load displacement mode
data = np.loadtxt(mode_file)
mode_vector = data[:, mode_column]
mode = mode_vector.reshape((len(atoms), 3))  # (N_atoms, 3)

# Step 3: Get symmetry operations from spglib
cell = (lattice, positions, numbers)
sym_ops = spglib.get_symmetry(cell, symprec=1e-3)
rotations = sym_ops['rotations']
translations = sym_ops['translations']

# Step 4: Determine how atoms are mapped under symmetry operations
def get_atom_mapping(positions, rotations, translations):
    mappings = []
    for R, t in zip(rotations, translations):
        new_pos = np.dot(positions, R.T) + t
        new_pos %= 1.0
        d = cdist(new_pos, positions)
        mapping = np.argmin(d, axis=1)
        mappings.append(mapping)
    return mappings

mappings = get_atom_mapping(positions, rotations, translations)

print(f"Number of symmetry operations: {len(rotations)}")
for i, R in enumerate(rotations):
    print(f"Op {i}: rotation=\n{R}")


# Step 5: Apply symmetry operations to the mode
def apply_symmetry_to_mode(mode, rotations, mappings):
    sym_modes = []
    for R, perm in zip(rotations, mappings):
        transformed = mode[perm] @ R
        sym_modes.append(transformed)
    return sym_modes

sym_modes = apply_symmetry_to_mode(mode, rotations, mappings)

# Step 6: Define characters for Ag and Bg of C2h (point group of SG 14)
characters = {
    'Ag': [1,  1,  1,  1],
    'Bg': [1, -1,  1, -1],
}

# Step 7: Symmetrize using projection operator
if target_irrep not in characters:
    raise ValueError(f"Unknown irrep: {target_irrep}")

char_list = characters[target_irrep]
# Repeat characters to match all symmetry operations (including translations)
irrep_chars = list(itertools.islice(itertools.cycle(char_list), len(rotations)))

def project_mode(mode, sym_modes, irrep_chars):
    G = len(sym_modes)
    projection = np.zeros_like(mode)
    for c, m in zip(irrep_chars, sym_modes):
        projection += c * m
    return projection / G

symmetrized_mode = project_mode(mode, sym_modes, irrep_chars)

# Measure deviation from ideal symmetry
diff = mode - symmetrized_mode
deviation = np.linalg.norm(diff) / np.linalg.norm(mode)
print(f"Symmetry deviation (relative L2 norm): {deviation:.6f}")

# Step 8: Save result
output_file = f"mode_{target_irrep}_symmetrized.txt"
np.savetxt(output_file, symmetrized_mode.reshape(-1))
print(f"✅ Symmetrized mode saved to: {output_file}")

