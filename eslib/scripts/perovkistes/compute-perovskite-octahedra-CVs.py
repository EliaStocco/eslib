#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.io import iread, write
from ase.neighborlist import neighbor_list
from eslib.formatting import esfmt
from eslib.functions import unique_elements

#---------------------------------------#
description = "Compute collective variables for the oxygen octahedra in a perovskite trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="input trajectory with unit-cell labels")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=str, help="frames to process (default: %(default)s)", default=":")
    parser.add_argument("-g" , "--aggregation" , **argv, required=False, type=str, help="per-atom aggregation array copied to the output (default: %(default)s)", default="unit-cell")
    parser.add_argument("-c" , "--cutoff"      , **argv, required=False, type=float, help="Ti-O neighbour cutoff in angstrom (default: %(default)s)", default=3.0)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output extxyz trajectory (default: %(default)s)", default="perovskite-octahedra-CVs.extxyz")
    return parser

#---------------------------------------#
def get_b_symbol(atoms:Atoms):
    symbols = np.asarray(atoms.get_chemical_symbols())
    non_oxygen = symbols[symbols != "O"]
    site_symbols, _, _ = unique_elements(non_oxygen)
    if len(site_symbols) != 2:
        raise ValueError(
            "Removing oxygen atoms must leave exactly two chemical species; "
            f"found {site_symbols}"
        )
    return site_symbols[1]

#---------------------------------------#
def get_lattice_axes(atoms:Atoms):
    """Return a right-handed orthonormal frame aligned with the cell vectors."""
    cell = atoms.cell.array
    axes, _ = np.linalg.qr(cell.T)
    axes = axes.T
    for n in range(3):
        if np.dot(axes[n], cell[n]) < 0:
            axes[n] *= -1
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes

#---------------------------------------#
def get_octahedra(atoms:Atoms, b_symbol:str, cutoff:float):
    """Find the six nearest O atoms around each B-site atom under PBC."""
    if cutoff <= 0:
        raise ValueError("--cutoff must be positive")

    symbols = np.asarray(atoms.get_chemical_symbols())
    b_indices = np.where(symbols == b_symbol)[0]
    if len(b_indices) == 0:
        raise ValueError(f"No '{b_symbol}' atoms were found")

    first, second, distance, vector = neighbor_list(
        "ijdD", atoms, cutoff, self_interaction=False
    )
    mask = (symbols[first] == b_symbol) & (symbols[second] == "O")
    first = first[mask]
    second = second[mask]
    distance = distance[mask]
    vector = vector[mask]

    order = np.argsort(first)
    first = first[order]
    second = second[order]
    distance = distance[order]
    vector = vector[order]
    count = np.bincount(first, minlength=len(atoms))[b_indices]
    if not np.all(count == 6):
        values, multiplicity = np.unique(count, return_counts=True)
        raise ValueError(
            f"Each '{b_symbol}' atom must have exactly six O neighbours within "
            f"{cutoff:g} angstrom; found {dict(zip(values, multiplicity))}"
        )
    if not np.array_equal(first.reshape(-1, 6)[:, 0], b_indices):
        raise ValueError("Internal error while grouping the oxygen neighbours")

    return b_indices, second.reshape(-1, 6), distance.reshape(-1, 6), vector.reshape(-1, 6, 3)

#---------------------------------------#
def get_octahedral_CVs(atoms:Atoms, b_indices:np.ndarray, oxygen_vectors:np.ndarray, distances:np.ndarray):
    """Compute off-centering, deformation, and orientation collective variables."""
    n_octahedra = len(b_indices)
    directions = oxygen_vectors / distances[:, :, np.newaxis]
    oxygen_center = atoms.positions[b_indices] + np.mean(oxygen_vectors, axis=1)
    offcentering_vector = -np.mean(oxygen_vectors, axis=1)
    offcentering_distance = np.linalg.norm(offcentering_vector, axis=1)

    mean_distance = np.mean(distances, axis=1)
    distance_std = np.std(distances, axis=1)
    bond_distortion = np.mean(
        ((distances - mean_distance[:, np.newaxis]) / mean_distance[:, np.newaxis]) ** 2,
        axis=1,
    )

    pair_i, pair_j = np.triu_indices(6, k=1)
    cosine = np.einsum("nij,nkj->nik", directions, directions)
    angles = np.degrees(np.arccos(np.clip(cosine[:, pair_i, pair_j], -1, 1)))
    ideal_angle = np.where(angles > 135, 180, 90)
    angle_distortion = np.sqrt(np.mean((angles - ideal_angle) ** 2, axis=1))

    lattice_axes = get_lattice_axes(atoms)
    projection = np.einsum("noi,ji->noj", directions, lattice_axes)
    positive = np.argmax(projection, axis=1)
    negative = np.argmin(projection, axis=1)
    assigned = np.concatenate((positive, negative), axis=1)
    if any(len(np.unique(assigned[n])) != 6 for n in range(n_octahedra)):
        raise ValueError(
            "Could not assign the six oxygen neighbours to three opposite axes. "
            "The octahedra are too distorted for the cell-aligned tilt definition."
        )

    index = np.arange(n_octahedra)[:, np.newaxis]
    octahedral_axes = directions[index, positive] - directions[index, negative]
    octahedral_axes /= np.linalg.norm(octahedral_axes, axis=2)[:, :, np.newaxis]
    axis_tilt = np.degrees(np.arccos(np.clip(
        np.einsum("nij,ij->ni", octahedral_axes, lattice_axes), -1, 1
    )))

    # Kabsch rotation: the best rigid rotation of the cell frame onto the
    # three local octahedral axes.  Its angle is a scalar local tilt measure.
    cross_covariance = np.einsum("ji,njk->nik", lattice_axes, octahedral_axes)
    left, _, right = np.linalg.svd(cross_covariance)
    rotation = np.matmul(left, right)
    negative_determinant = np.linalg.det(rotation) < 0
    if np.any(negative_determinant):
        left[negative_determinant, :, -1] *= -1
        rotation = np.matmul(left, right)
    octahedral_tilt = np.degrees(np.arccos(np.clip(
        (np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1, 1
    )))

    return {
        "octahedron-center": oxygen_center,
        "Ti-offcentering-vector": offcentering_vector,
        "Ti-offcentering-distance": offcentering_distance,
        "Ti-O-distance": distances,
        "Ti-O-distance-mean": mean_distance,
        "Ti-O-distance-std": distance_std,
        "octahedron-bond-distortion": bond_distortion,
        "octahedron-angle-distortion": angle_distortion,
        "octahedron-axis": octahedral_axes,
        "octahedron-axis-tilt": axis_tilt,
        "octahedron-tilt": octahedral_tilt,
    }

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    read_kwargs = {"index": args.index}
    if args.input_format is not None:
        read_kwargs["format"] = args.input_format

    print("\tReading and processing atomic structures from file '{:s}' ...".format(args.input))
    output_written = False
    reference_aggregation = None
    reference_b_indices = None

    for frame, atoms in enumerate(iread(args.input, **read_kwargs)):
        b_symbol = get_b_symbol(atoms)
        b_indices, oxygen_indices, distances, oxygen_vectors = get_octahedra(
            atoms, b_symbol, args.cutoff
        )
        if args.aggregation not in atoms.arrays:
            raise ValueError(f"The input does not contain '{args.aggregation}' in atoms.arrays")
        aggregation = np.asarray(atoms.arrays[args.aggregation])[b_indices]

        if reference_aggregation is None:
            reference_aggregation = aggregation.copy()
            reference_b_indices = b_indices.copy()
            print("\tFound {:d} {}O6 octahedra.".format(len(b_indices), b_symbol))
        else:
            if not np.array_equal(aggregation, reference_aggregation):
                raise ValueError(f"The '{args.aggregation}' assignment changes in frame {frame}")
            if not np.array_equal(b_indices, reference_b_indices):
                raise ValueError(f"The B-site atom ordering changes in frame {frame}")

        collective_variables = get_octahedral_CVs(
            atoms, b_indices, oxygen_vectors, distances
        )
        output = Atoms(
            symbols=[b_symbol] * len(b_indices),
            positions=atoms.positions[b_indices],
            cell=atoms.cell,
            pbc=atoms.pbc,
            info=atoms.info.copy(),
        )
        output.new_array(args.aggregation, aggregation)
        output.new_array("B-index", b_indices)
        output.new_array("O-index", oxygen_indices)
        output.new_array("Ti-O-vector", oxygen_vectors.reshape(len(b_indices), -1))
        for name, value in collective_variables.items():
            if value.ndim > 2:
                value = value.reshape(len(b_indices), -1)
            output.new_array(name, value)
        output.info["frame"] = frame

        write(args.output, output, format="extxyz", append=output_written)
        output_written = True
        if (frame + 1) % 100 == 0:
            print("\tProcessed {:d} frames.".format(frame + 1))

    if not output_written:
        raise ValueError("No structures were selected from the input trajectory")

    print("\tSaved {:d} frames to file '{:s}'.".format(frame + 1, args.output))

#---------------------------------------#
if __name__ == "__main__":
    main()
