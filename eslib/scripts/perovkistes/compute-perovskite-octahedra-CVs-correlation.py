#!/usr/bin/env python
import warnings

import numpy as np
from ase.io import iread
from eslib.formatting import esfmt, float_format
from eslib.input import str2bool

#---------------------------------------#
description = "Compute spatial correlations of perovskite oxygen-octahedron collective variables."

scalar_CVs = {
    "octahedron-bond-distortion": "bond_distortion_correlation",
    "octahedron-angle-distortion": "angle_distortion_correlation",
    "octahedron-tilt": "tilt_correlation",
}

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="extxyz produced by compute-perovskite-octahedra-CVs.py")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=str, help="frames to process (default: %(default)s)", default=":")
    parser.add_argument("-p" , "--periodic"    , **argv, required=False, type=str2bool, help="apply periodic boundary conditions to the unit-cell grid (default: %(default)s)", default=True)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output correlation table (default: %(default)s)", default="perovskite-octahedra-CVs-correlation.txt")
    return parser

#---------------------------------------#
def get_unit_cell_coordinates(atoms):
    if "unit-cell" not in atoms.arrays:
        raise ValueError("The input does not contain the 'unit-cell' array")

    unit_cell = np.asarray(atoms.arrays["unit-cell"])
    if unit_cell.ndim != 2 or unit_cell.shape[1] < 4:
        raise ValueError(
            "'unit-cell' must contain the aggregate index followed by three "
            f"grid coordinates; got shape {unit_cell.shape}"
        )

    coordinates = unit_cell[:, 1:4]
    if not np.allclose(coordinates, np.rint(coordinates)):
        raise ValueError("The unit-cell coordinates must be integers")
    return np.rint(coordinates).astype(int)

#---------------------------------------#
def prepare_pairs(coordinates:np.ndarray, periodic:bool):
    """Group all unique unit-cell pairs by their lattice-index distance."""
    n_unit_cells = len(coordinates)
    _, multiplicity = np.unique(coordinates, axis=0, return_counts=True)
    if np.any(multiplicity != 1):
        raise ValueError(
            "Each unit cell must have a unique three-dimensional grid index. "
            f"Found {np.count_nonzero(multiplicity != 1)} duplicated grid indices."
        )

    periods = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    if periodic and int(np.prod(periods)) != n_unit_cells:
        raise ValueError("Periodic correlations require a complete rectangular unit-cell grid")

    first, second = np.triu_indices(n_unit_cells, k=1)
    displacement = coordinates[second] - coordinates[first]
    if periodic:
        displacement -= periods * np.rint(displacement / periods).astype(int)

    distance_squared = np.einsum("ij,ij->i", displacement, displacement)
    unique_distance_squared, pair_bins = np.unique(distance_squared, return_inverse=True)
    pairs_per_distance = np.bincount(pair_bins, minlength=len(unique_distance_squared))
    return first, second, unique_distance_squared, pair_bins, pairs_per_distance

#---------------------------------------#
def get_CVs(atoms, frame:int):
    required = ["Ti-offcentering-vector", *scalar_CVs]
    missing = [name for name in required if name not in atoms.arrays]
    if missing:
        raise ValueError(
            "The input is missing collective variables: " + ", ".join(missing)
        )

    offcentering = np.asarray(atoms.arrays["Ti-offcentering-vector"], dtype=float)
    if offcentering.ndim != 2 or offcentering.shape[1] != 3:
        raise ValueError(
            f"'Ti-offcentering-vector' in frame {frame} must have shape "
            f"(n_octahedra, 3); got {offcentering.shape}"
        )

    scalar = {}
    for name in scalar_CVs:
        value = np.asarray(atoms.arrays[name], dtype=float)
        if value.ndim != 1 or len(value) != len(offcentering):
            raise ValueError(
                f"'{name}' in frame {frame} must have shape "
                f"({len(offcentering)},); got {value.shape}"
            )
        scalar[name] = value
    return offcentering, scalar

#---------------------------------------#
def connected_correlation(raw:np.ndarray, mean:float, variance:float):
    if variance <= np.finfo(float).eps:
        return np.full_like(raw, np.nan)
    return (raw - mean ** 2) / variance

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    read_kwargs = {"index": args.index}
    if args.input_format is not None:
        read_kwargs["format"] = args.input_format

    print("\tReading octahedral collective variables from file '{:s}' ...".format(args.input))
    reference_coordinates = None
    first = None
    second = None
    pair_bins = None
    pairs_per_distance = None
    distance_squared = None
    vector_correlation_sum = None
    angular_correlation_sum = None
    scalar_correlation_sum = None
    vector_sum = np.zeros(3)
    vector_squared_sum = 0.0
    scalar_sum = {name: 0.0 for name in scalar_CVs}
    scalar_squared_sum = {name: 0.0 for name in scalar_CVs}
    vector_correlation_frames = []
    angular_correlation_frames = []
    scalar_correlation_frames = {name: [] for name in scalar_CVs}
    n_observations = 0
    n_frames = 0

    for frame, atoms in enumerate(iread(args.input, **read_kwargs)):
        coordinates = get_unit_cell_coordinates(atoms)
        offcentering, scalar = get_CVs(atoms, frame)
        if len(coordinates) != len(offcentering):
            raise ValueError(
                f"Frame {frame} has {len(coordinates)} unit-cell coordinates but "
                f"{len(offcentering)} octahedra"
            )

        if reference_coordinates is None:
            reference_coordinates = coordinates.copy()
            first, second, distance_squared, pair_bins, pairs_per_distance = \
                prepare_pairs(coordinates, args.periodic)
            vector_correlation_sum = np.zeros(len(distance_squared) + 1)
            angular_correlation_sum = np.zeros(len(distance_squared) + 1)
            scalar_correlation_sum = {
                name: np.zeros(len(distance_squared) + 1) for name in scalar_CVs
            }
            print("\tFound {:d} octahedra and {:d} distinct nonzero grid distances.".format(
                len(coordinates), len(distance_squared)
            ))
        elif not np.array_equal(coordinates, reference_coordinates):
            raise ValueError(f"The unit-cell grid changes in frame {frame}")

        vector_sum += np.sum(offcentering, axis=0)
        vector_squared_sum += np.einsum("ij,ij->", offcentering, offcentering)
        vector_correlation_sum[0] += np.einsum("ij,ij->", offcentering, offcentering)
        vector_correlation_sum[1:] += np.bincount(
            pair_bins,
            weights=np.einsum("ij,ij->i", offcentering[first], offcentering[second]),
            minlength=len(distance_squared),
        )

        magnitude = np.linalg.norm(offcentering, axis=1)
        zero_magnitude = magnitude <= np.finfo(float).eps
        if np.any(zero_magnitude):
            warnings.warn(
                f"Frame {frame} contains {np.count_nonzero(zero_magnitude)} zero "
                "Ti off-centering vector(s); their angular correlation is undefined "
                "and will be written as NaN",
                RuntimeWarning,
            )
        direction = np.full_like(offcentering, np.nan)
        direction[~zero_magnitude] = (
            offcentering[~zero_magnitude]
            / magnitude[~zero_magnitude, np.newaxis]
        )
        angular_correlation_sum[0] += np.nan if np.any(zero_magnitude) else len(direction)
        angular_correlation_sum[1:] += np.bincount(
            pair_bins,
            weights=np.einsum("ij,ij->i", direction[first], direction[second]),
            minlength=len(distance_squared),
        )

        for name, value in scalar.items():
            scalar_sum[name] += np.sum(value)
            scalar_squared_sum[name] += np.dot(value, value)
            scalar_correlation_sum[name][0] += np.dot(value, value)
            scalar_correlation_sum[name][1:] += np.bincount(
                pair_bins,
                weights=value[first] * value[second],
                minlength=len(distance_squared),
            )

        pairs = np.concatenate(([len(offcentering)], pairs_per_distance))
        frame_vector_correlation = np.empty(len(pairs))
        frame_vector_correlation[0] = np.einsum("ij,ij->", offcentering, offcentering)
        frame_vector_correlation[1:] = np.bincount(
            pair_bins,
            weights=np.einsum("ij,ij->i", offcentering[first], offcentering[second]),
            minlength=len(distance_squared),
        )
        vector_correlation_frames.append(frame_vector_correlation / pairs)

        frame_angular_correlation = np.empty(len(pairs))
        frame_angular_correlation[0] = np.nan if np.any(zero_magnitude) else len(direction)
        frame_angular_correlation[1:] = np.bincount(
            pair_bins,
            weights=np.einsum("ij,ij->i", direction[first], direction[second]),
            minlength=len(distance_squared),
        )
        angular_correlation_frames.append(frame_angular_correlation / pairs)

        for name, value in scalar.items():
            frame_scalar_correlation = np.empty(len(pairs))
            frame_scalar_correlation[0] = np.dot(value, value)
            frame_scalar_correlation[1:] = np.bincount(
                pair_bins,
                weights=value[first] * value[second],
                minlength=len(distance_squared),
            )
            scalar_correlation_frames[name].append(frame_scalar_correlation / pairs)

        n_observations += len(offcentering)
        n_frames += 1
        if n_frames % 100 == 0:
            print("\tProcessed {:d} frames.".format(n_frames))

    if n_frames == 0:
        raise ValueError("No structures were selected from the input trajectory")

    # For a complete periodic grid every site occurs equally often in each
    # distance shell, so subtracting the global mean gives the connected
    # scalar/vector correlation in each shell.
    samples = np.concatenate(([len(reference_coordinates)], pairs_per_distance)) * n_frames
    vector_raw = vector_correlation_sum / samples
    vector_mean = vector_sum / n_observations
    vector_variance = vector_squared_sum / n_observations - np.dot(vector_mean, vector_mean)
    offcentering_correlation = connected_correlation(
        vector_raw, 0.0, vector_variance
    )
    offcentering_correlation -= np.dot(vector_mean, vector_mean) / vector_variance
    angular_correlation = angular_correlation_sum / samples
    offcentering_correlation_frames = (
        np.asarray(vector_correlation_frames) - np.dot(vector_mean, vector_mean)
    ) / vector_variance
    offcentering_correlation_std = np.std(
        offcentering_correlation_frames, axis=0, ddof=1 if n_frames > 1 else 0
    )
    angular_correlation_std = np.std(
        angular_correlation_frames, axis=0, ddof=1 if n_frames > 1 else 0
    )

    correlation = {}
    correlation_std = {}
    for name, output_name in scalar_CVs.items():
        raw = scalar_correlation_sum[name] / samples
        mean = scalar_sum[name] / n_observations
        variance = scalar_squared_sum[name] / n_observations - mean ** 2
        correlation[output_name] = connected_correlation(raw, mean, variance)
        correlation_std[output_name] = np.std(
            connected_correlation(
                np.asarray(scalar_correlation_frames[name]), mean, variance
            ),
            axis=0,
            ddof=1 if n_frames > 1 else 0,
        )

    distance = np.sqrt(np.concatenate(([0], distance_squared)))
    output = np.column_stack((
        distance,
        offcentering_correlation,
        offcentering_correlation_std,
        angular_correlation,
        angular_correlation_std,
        *(
            value for key in correlation for value in (correlation[key], correlation_std[key])
        ),
        samples,
    ))

    #------------------#
    print("\tSaving correlation table to file '{:s}' ... ".format(args.output), end="")
    header = " ".join((
        "distance_in_unit_cells",
        "offcentering_correlation",
        "offcentering_correlation_std",
        "offcentering_angular_correlation",
        "offcentering_angular_correlation_std",
        *(
            value for key in correlation for value in (key, f"{key}_std")
        ),
        "n_samples",
    ))
    np.savetxt(
        args.output,
        output,
        fmt=[float_format] * (output.shape[1] - 1) + ["%d"],
        header=header,
        comments="",
    )
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
