#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.cell import Cell
from eslib.formatting import esfmt
from eslib.input import flist
from eslib.tools import frac2cart, cart2frac

#---------------------------------------#
# Description of the script's purpose
description = "Fill a box with N water molecules (random orientations, correct bond lengths/angles)."

documentation = \
"https://en.wikipedia.org/wiki/Lloyd%27s_algorithm\n\
https://en.wikipedia.org/wiki/Halton_sequence"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-n" , "--n_molecules"  , **argv, required=True , type=int, help="number of water molecules")
    parser.add_argument("-c" , "--cell"         , **argv, required=True , type=flist, help="a, b, c, α, β, γ [ang,deg] or 9 matrix elements")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file with the atomic structures")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
def halton_sequence(size, dim=3):
    """Generate a Halton low-discrepancy sequence in [0,1)^dim."""
    def halton_single(index, base):
        f, r, i = 1.0, 0.0, index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    bases = [2,3,5]
    seq = np.zeros((size, dim))
    for d in range(dim):
        for i in range(size):
            seq[i,d] = halton_single(i+1, bases[d % len(bases)])
    return seq

def lloyd_iteration(F, cell):
    """One Lloyd iteration using periodic Voronoi (approx with 3x3x3 tiling)."""
    from scipy.spatial import Voronoi
    N = len(F)
    # lattice = cell.array
    shifts = np.array([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]])
    F_super = (F[None,:,:] + shifts[:,None,:]).reshape(-1,3)
    R_super = frac2cart(cell,F_super)
    vor = Voronoi(R_super)

    new_F = np.copy(F)
    for i in range(N):
        idx = np.where(np.all(np.isclose(F_super, F[i], atol=1e-8), axis=1))[0]
        if len(idx)==0: 
            continue
        region_index = vor.point_region[idx[0]]
        verts = vor.regions[region_index]
        if -1 in verts or len(verts)==0:
            continue
        centroid_cart = np.mean(vor.vertices[verts], axis=0)
        new_F[i] = cart2frac(cell,centroid_cart)
    return new_F % 1.0

def lloyd_relax(F, cell, n_iter=5):
    for _ in range(n_iter):
        F = lloyd_iteration(F, cell)
    return F

#---------------------------------------#
# Water builder utilities
def random_rotation_matrix():
    """Uniform random rotation in 3D."""
    q = np.random.normal(size=4)
    q /= np.linalg.norm(q)
    w,x,y,z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return R

def build_water_geometry():
    """Return coordinates of one water molecule in local frame (O at origin)."""
    d_OH = 0.9572  # Å
    angle_HOH = np.deg2rad(104.52)

    # place O at origin, one H along +x
    O = np.array([0.0, 0.0, 0.0])
    H1 = np.array([d_OH, 0.0, 0.0])
    # rotate around z-axis to place H2
    H2 = np.array([d_OH*np.cos(angle_HOH), d_OH*np.sin(angle_HOH), 0.0])

    return np.vstack([O,H1,H2])

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):
    print("\tConstructing the cell ... ", end="")
    if len(args.cell) in [3,6]:
        cell = Cell.fromcellpar(args.cell)
    elif len(args.cell) == 9:
        cell = np.asarray(args.cell).reshape((3,3))
        cell = Cell(cell)
    else:
        raise ValueError("Cell must have 3, 6, or 9 parameters.")
    print("done")

    print("\tcellpar: ", cell.cellpar())

    N = args.n_molecules

    #------------------#
    print(f"\tPlacing {N} water molecules using Halton + Lloyd relaxation ... ", end="")
    F0 = halton_sequence(N)
    F_relaxed = lloyd_relax(F0, cell, n_iter=5)
    O_positions = frac2cart(cell,F_relaxed)
    print("done")

    #------------------#
    
    symbols = []
    positions = []

    water_local = build_water_geometry()

    for O_pos in O_positions:
        R = random_rotation_matrix()
        water_rot = water_local @ R.T
        water_cart = water_rot + O_pos
        symbols.extend(["O","H","H"])
        positions.extend(water_cart)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    #------------------#
    print(f"\tWriting {args.output} ... ", end="")
    atoms.write(args.output, format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
