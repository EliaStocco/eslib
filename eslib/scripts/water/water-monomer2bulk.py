#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.neighborlist import neighbor_list
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Generate a structure of N water molecules randomly placed and rotated in a given box, then analyze distances."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b",}
    parser.add_argument("-i", "--input", **argv, type=str, required=True,
                        help="Input file with single water monomer [extxyz, xyz, etc.]")
    parser.add_argument("-if", "--input_format", **argv, type=str, required=False,
                        help="Input file format (default: %(default)s)", default=None)
    parser.add_argument("-n", "--nmol", **argv, type=int, required=True,
                        help="Number of water molecules to generate")
    parser.add_argument("-b", "--box", **argv, type=float, nargs=3, required=True,
                        help="Box dimensions (Lx Ly Lz) in Å")
    parser.add_argument("-o", "--output", **argv, type=str, required=True,
                        help="Output structure file (extxyz, xyz, etc.)")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False,
                        help="Output file format (default: %(default)s)", default=None)
    parser.add_argument("-d", "--min_dist", **argv, type=float, required=False,
                        help="Minimum distance between molecule centers [Å] (default: %(default)s)", default=2.0)
    parser.add_argument("--violin", **argv, type=str, required=False,
                        help="Output violin plot file (default: %(default)s)", default="distances_violin.png")
    return parser

#---------------------------------------#
def random_rotation_matrix():
    rand = np.random.rand(3)
    theta = rand[0] * 2 * np.pi
    phi = rand[1] * 2 * np.pi
    z = rand[2] * 2.0 - 1.0
    r = np.sqrt(1.0 - z*z)
    axis = np.array([r*np.cos(phi), r*np.sin(phi), z])
    w = np.cos(theta / 2.0)
    x, y, z = axis * np.sin(theta / 2.0)
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

#---------------------------------------#
def pbc_distance(vec, box):
    return vec - box * np.round(vec / box)

#---------------------------------------#
def compute_distances(atoms):
    """Compute O-O, H-H, and H-O distances with PBC."""
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()
    box = np.array(cell.lengths())

    # Get all pairs within half the smallest box length (safe max cutoff)
    cutoff = min(box) / 2.0
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff)

    distances = {"O-O": [], "H-H": [], "H-O": []}
    for i, j, d in zip(i_list, j_list, d_list):
        if i >= j:
            continue  # avoid double counting
        si, sj = symbols[i], symbols[j]
        if si == "O" and sj == "O":
            distances["O-O"].append(d)
        elif si == "H" and sj == "H":
            distances["H-H"].append(d)
        elif (si == "O" and sj == "H") or (si == "H" and sj == "O"):
            distances["H-O"].append(d)

    return distances

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    # Load monomer
    print(f"\n\tReading water monomer from '{args.input}' ... ", end="")
    monomer = AtomicStructures.from_file(file=args.input, format=args.input_format)
    if len(monomer) != 1:
        raise ValueError("Input file must contain exactly ONE water molecule.")
    monomer = monomer[0]
    print("done")

    Lx, Ly, Lz = args.box
    box = np.array([Lx, Ly, Lz])
    print(f"\tTarget box dimensions: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} Å")

    final_structure = Atoms()
    mol_centers = []

    # Place molecules randomly
    for i in range(args.nmol):
        placed = False
        while not placed:
            center = np.random.rand(3) * box
            too_close = False
            for prev_center in mol_centers:
                disp = pbc_distance(center - prev_center, box)
                dist = np.linalg.norm(disp)
                if dist < args.min_dist:
                    too_close = True
                    break
            if not too_close:
                placed = True
                mol_centers.append(center)
                mol_copy = monomer.copy()
                R = random_rotation_matrix()
                mol_copy.positions = (mol_copy.positions - mol_copy.get_center_of_mass()) @ R.T
                mol_copy.translate(center)
                final_structure += mol_copy

    final_structure.set_cell(box)
    final_structure.pbc = True

    # Save output
    print(f"\n\tWriting structure to '{args.output}' ... ", end="")
    AtomicStructures([final_structure]).to_file(file=args.output, format=args.output_format)
    print("done")

    # Compute distances
    print("\n\tComputing pairwise distances ... ", end="")
    distances = compute_distances(final_structure)
    print("done")

    # Print stats
    for pair_type, vals in distances.items():
        vals = np.array(vals)
        print(f"{pair_type}: min={vals.min():.3f} Å, mean={vals.mean():.3f} Å, max={vals.max():.3f} Å")

    # Prepare violin plot
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [distances["O-O"], distances["H-H"], distances["H-O"]]
    labels = ["O-O", "H-H", "H-O"]
    vp = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=False)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Distance (Å)")
    ax.set_title("Pairwise distances in generated structure")
    plt.tight_layout()
    plt.savefig(args.violin, dpi=300)
    print(f"\n\tViolin plot saved to '{args.violin}'")

if __name__ == "__main__":
    main()
