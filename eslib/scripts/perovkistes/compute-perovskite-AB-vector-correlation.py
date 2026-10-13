#!/usr/bin/env python
import warnings

import numpy as np
from ase.io import iread
from eslib.formatting import esfmt, float_format
from eslib.input import str2bool

#---------------------------------------#
description = "Compute radial and direction-resolved spatial correlations of perovskite A-B vectors."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="extxyz produced by compute-perovskite-AB-vector.py")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=str, help="frames to process (default: %(default)s)", default=":")
    parser.add_argument("-p" , "--periodic"    , **argv, required=False, type=str2bool, help="apply periodic boundary conditions to the unit-cell grid (default: %(default)s)", default=True)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output correlation table (default: %(default)s)", default="perovskite-AB-vector-correlation.txt")
    parser.add_argument("-do", "--directional_output", **argv, required=False, type=str, help="output direction-resolved angular-correlation table (default: %(default)s)", default="perovskite-AB-vector-directional-correlation.txt")
    parser.add_argument("-go", "--gaussian_output", **argv, required=False, type=str, help="output A-B-vector norm Gaussian-membership table (default: %(default)s)", default="perovskite-AB-vector-gaussian-membership.txt")
    parser.add_argument("-gc", "--gaussian_components", **argv, required=False, type=int, help="number of Gaussian components; infer it with BIC when omitted (default: %(default)s)", default=None)
    parser.add_argument("-mg", "--max_gaussian_components", **argv, required=False, type=int, help="largest number of Gaussian components considered by BIC (default: %(default)s)", default=8)
    return parser

#---------------------------------------#
def get_unit_cell_coordinates(atoms):
    if "unit-cell" not in atoms.arrays:
        raise ValueError(
            "The input does not contain the 'unit-cell' array. It must be the "
            "output of compute-perovskite-AB-vector.py."
        )

    unit_cell = np.asarray(atoms.arrays["unit-cell"])
    if unit_cell.ndim != 2 or unit_cell.shape[1] < 4:
        raise ValueError(
            "'unit-cell' must contain the aggregate index followed by its three "
            f"grid coordinates; got shape {unit_cell.shape}"
        )

    coordinates = unit_cell[:, 1:4]
    if not np.allclose(coordinates, np.rint(coordinates)):
        raise ValueError("The unit-cell coordinates must be integers")
    return np.rint(coordinates).astype(int)

#---------------------------------------#
def get_vector(atoms, frame:int):
    if "AB-vector" not in atoms.arrays:
        raise ValueError(
            "The input does not contain the 'AB-vector' array. It must be the "
            "output of compute-perovskite-AB-vector.py."
        )

    vector = np.asarray(atoms.arrays["AB-vector"], dtype=float)
    if vector.ndim != 2 or vector.shape[1] != 3:
        raise ValueError(
            f"'AB-vector' in frame {frame} must have shape (n_unit_cells, 3); "
            f"got {vector.shape}"
        )
    return vector

#---------------------------------------#
def prepare_pairs(coordinates:np.ndarray, periodic:bool):
    """Group all unique unit-cell pairs by their grid distance."""
    n_unit_cells = len(coordinates)
    _, multiplicity = np.unique(coordinates, axis=0, return_counts=True)
    if np.any(multiplicity != 1):
        raise ValueError(
            "Each unit cell must have a unique three-dimensional grid index. "
            f"Found {np.count_nonzero(multiplicity != 1)} duplicated grid indices."
        )

    first, second = np.triu_indices(n_unit_cells, k=1)
    displacement = coordinates[second] - coordinates[first]

    if periodic:
        periods = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        displacement -= periods * np.rint(displacement / periods).astype(int)

    distance_squared = np.einsum("ij,ij->i", displacement, displacement)
    unique_distance_squared, pair_bins = np.unique(distance_squared, return_inverse=True)
    pairs_per_distance = np.bincount(pair_bins, minlength=len(unique_distance_squared))

    return first, second, unique_distance_squared, pair_bins, pairs_per_distance

#---------------------------------------#
def prepare_directional_correlation(coordinates:np.ndarray):
    """Prepare a periodic grid for C(Delta n) on every lattice displacement."""
    origin = coordinates.min(axis=0)
    shape = coordinates.max(axis=0) - origin + 1
    relative_coordinates = coordinates - origin

    if np.any(relative_coordinates < 0):
        raise ValueError("The unit-cell grid contains invalid coordinates")
    if len(np.unique(relative_coordinates, axis=0)) != len(coordinates):
        raise ValueError("Each unit cell must have a unique three-dimensional grid index")
    if len(coordinates) != int(np.prod(shape)):
        raise ValueError(
            "A direction-resolved periodic correlation requires a complete "
            "rectangular unit-cell grid"
        )

    atom_index = np.empty(shape, dtype=int)
    atom_index[tuple(relative_coordinates.T)] = np.arange(len(coordinates))
    shifts = [np.rint(np.fft.fftfreq(size) * size).astype(int) for size in shape]
    displacement = np.stack(np.meshgrid(*shifts, indexing="ij"), axis=-1)
    return tuple(shape), atom_index, displacement

#---------------------------------------#
def directional_angular_correlation(unit_vector:np.ndarray, shape, atom_index):
    """Return <u(n) dot u(n+Delta n)> for all periodic grid displacements."""
    direction = unit_vector[atom_index]

    correlation = np.zeros(shape, dtype=float)
    for component in range(3):
        transform = np.fft.fftn(direction[..., component])
        correlation += np.fft.ifftn(transform.conjugate() * transform).real
    return correlation / np.prod(shape)

#---------------------------------------#
def gaussian_cluster_means(values:np.ndarray, n_components, max_components:int):
    """Assign each finite value to a 1D Gaussian and return its component mean."""
    from sklearn.mixture import GaussianMixture

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    component_mean = np.full(len(values), np.nan)
    if not np.any(finite):
        return component_mean, np.empty(0)

    samples = values[finite].reshape(-1, 1)
    n_unique = len(np.unique(samples))
    max_available = min(len(samples), n_unique)
    if n_components is None:
        n_components = min(max_components, max_available)
        models = [
            GaussianMixture(n_components=n, random_state=0).fit(samples)
            for n in range(1, n_components + 1)
        ]
        model = min(models, key=lambda candidate: candidate.bic(samples))
    else:
        if n_components > max_available:
            raise ValueError(
                f"--gaussian_components={n_components} requires at least that many "
                f"distinct finite vector norms; found {max_available}"
            )
        model = GaussianMixture(n_components=n_components, random_state=0).fit(samples)

    labels = model.predict(samples)
    means = model.means_.ravel()
    component_mean[finite] = means[labels]
    return component_mean, np.sort(means)

#---------------------------------------#
def get_source_frame(atoms, selected_frame:int):
    """Use the frame stored by compute-perovskite-AB-vector.py when available."""
    source_frame = atoms.info.get("frame", selected_frame)
    try:
        return int(source_frame)
    except (TypeError, ValueError):
        warnings.warn(
            f"Frame {selected_frame} has a non-integer 'frame' metadata value; "
            "using its index in the selected trajectory instead",
            RuntimeWarning,
        )
        return selected_frame

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.gaussian_components is not None and args.gaussian_components < 1:
        raise ValueError("--gaussian_components must be positive")
    if args.max_gaussian_components < 1:
        raise ValueError("--max_gaussian_components must be positive")

    read_kwargs = {"index": args.index}
    if args.input_format is not None:
        read_kwargs["format"] = args.input_format

    print("\tReading A-B vectors from file '{:s}' ...".format(args.input))
    reference_coordinates = None
    first = None
    second = None
    pair_bins = None
    pairs_per_distance = None
    distance_squared = None
    correlation_sum = None
    angular_correlation_sum = None
    normalized_correlation_frames = []
    angular_correlation_frames = []
    directional_angular_correlation_frames = []
    directional_shape = None
    directional_atom_index = None
    directional_displacement = None
    vector_norm_frames = []
    source_frames = []
    n_frames = 0

    if args.directional_output is not None and not args.periodic:
        raise ValueError(
            "Direction-resolved angular correlations require '--periodic true' "
            "because they are evaluated on a complete periodic grid"
        )

    for frame, atoms in enumerate(iread(args.input, **read_kwargs)):
        coordinates = get_unit_cell_coordinates(atoms)
        vector = get_vector(atoms, frame)
        if len(coordinates) != len(vector):
            raise ValueError(
                f"Frame {frame} has {len(coordinates)} unit-cell coordinates but "
                f"{len(vector)} A-B vectors"
            )

        if reference_coordinates is None:
            reference_coordinates = coordinates.copy()
            first, second, distance_squared, pair_bins, pairs_per_distance = \
                prepare_pairs(coordinates, args.periodic)
            correlation_sum = np.zeros(len(distance_squared) + 1)
            angular_correlation_sum = np.zeros(len(distance_squared) + 1)
            if args.directional_output is not None:
                directional_shape, directional_atom_index, directional_displacement = \
                    prepare_directional_correlation(coordinates)
            print("\tFound {:d} unit cells and {:d} distinct nonzero grid distances.".format(
                len(coordinates), len(distance_squared)
            ))
        elif not np.array_equal(coordinates, reference_coordinates):
            raise ValueError(
                f"The unit-cell grid changes in frame {frame}; a fixed grid is required"
            )

        correlation_sum[0] += np.einsum("ij,ij->", vector, vector)
        dot_product = np.einsum("ij,ij->i", vector[first], vector[second])
        correlation_sum[1:] += np.bincount(
            pair_bins,
            weights=dot_product,
            minlength=len(distance_squared),
        )

        magnitude = np.linalg.norm(vector, axis=1)
        vector_norm_frames.append(magnitude)
        source_frames.append(get_source_frame(atoms, frame))
        zero_magnitude = magnitude <= np.finfo(float).eps
        if np.any(zero_magnitude):
            warnings.warn(
                f"Frame {frame} contains {np.count_nonzero(zero_magnitude)} zero "
                "A-B vector(s); their angular correlation is undefined and will "
                "be written as NaN",
                RuntimeWarning,
            )
        unit_vector = np.full_like(vector, np.nan)
        unit_vector[~zero_magnitude] = (
            vector[~zero_magnitude] / magnitude[~zero_magnitude, np.newaxis]
        )
        angular_correlation_sum[0] += np.nan if np.any(zero_magnitude) else len(vector)
        angular_dot_product = np.einsum(
            "ij,ij->i", unit_vector[first], unit_vector[second]
        )
        angular_correlation_sum[1:] += np.bincount(
            pair_bins,
            weights=angular_dot_product,
            minlength=len(distance_squared),
        )

        pairs = np.concatenate(([len(vector)], pairs_per_distance))
        frame_correlation = np.empty(len(pairs))
        frame_correlation[0] = np.einsum("ij,ij->", vector, vector)
        frame_correlation[1:] = np.bincount(
            pair_bins,
            weights=dot_product,
            minlength=len(distance_squared),
        )
        frame_correlation /= pairs
        normalized_correlation_frames.append(frame_correlation / frame_correlation[0])

        frame_angular_correlation = np.empty(len(pairs))
        frame_angular_correlation[0] = np.nan if np.any(zero_magnitude) else len(vector)
        frame_angular_correlation[1:] = np.bincount(
            pair_bins,
            weights=angular_dot_product,
            minlength=len(distance_squared),
        )
        angular_correlation_frames.append(frame_angular_correlation / pairs)
        if args.directional_output is not None:
            directional_angular_correlation_frames.append(
                directional_angular_correlation(
                    unit_vector, directional_shape, directional_atom_index
                ).ravel()
            )
        n_frames += 1

        if n_frames % 100 == 0:
            print("\tProcessed {:d} frames.".format(n_frames))

    if n_frames == 0:
        raise ValueError("No structures were selected from the input trajectory")

    # Keep this diagnostic separate from the correlations: it maps every norm
    # to its unit cell and trajectory frame, using the fitted component mean as
    # the Gaussian label.
    vector_norm = np.concatenate(vector_norm_frames)
    gaussian_mean, gaussian_means = gaussian_cluster_means(
        vector_norm, args.gaussian_components, args.max_gaussian_components
    )
    if not np.all(np.isfinite(vector_norm)):
        warnings.warn(
            "Non-finite A-B vector norms cannot be assigned to a Gaussian and "
            "will be written with a NaN Gaussian label",
            RuntimeWarning,
        )
    diagnostic_frames = np.repeat(np.arange(n_frames), len(reference_coordinates))
    diagnostic_source_frames = np.repeat(source_frames, len(reference_coordinates))
    diagnostic_coordinates = np.tile(reference_coordinates, (n_frames, 1))
    gaussian_membership = np.column_stack((
        diagnostic_frames,
        diagnostic_source_frames,
        diagnostic_coordinates,
        vector_norm,
        gaussian_mean,
    ))

    print("\tFitted {:d} Gaussian component(s) with mean norm(s): {:s}".format(
        len(gaussian_means), ", ".join(f"{mean:.8g}" for mean in gaussian_means)
    ))
    print("\tSaving Gaussian-membership table to file '{:s}' ... ".format(
        args.gaussian_output
    ), end="")
    np.savetxt(
        args.gaussian_output,
        gaussian_membership,
        fmt=["%d"] * 5 + [float_format] * 2,
        header=(
            "selected_frame source_frame unit_cell_i unit_cell_j unit_cell_k "
            "AB_vector_norm_angstrom gaussian_mean_norm_angstrom"
        ),
        comments="",
    )
    print("done")

    # The r=0 value is the mean squared vector magnitude.  Each nonzero-r
    # average is over all unordered pairs at that grid distance and all frames.
    samples = np.concatenate(([len(reference_coordinates)], pairs_per_distance)) * n_frames
    correlation = correlation_sum / samples
    normalized_correlation = correlation / correlation[0]
    angular_correlation = angular_correlation_sum / samples
    normalized_correlation_std = np.std(
        normalized_correlation_frames, axis=0, ddof=1 if n_frames > 1 else 0
    )
    angular_correlation_std = np.std(
        angular_correlation_frames, axis=0, ddof=1 if n_frames > 1 else 0
    )
    distance = np.sqrt(np.concatenate(([0], distance_squared)))
    output = np.column_stack((
        distance,
        correlation,
        normalized_correlation,
        normalized_correlation_std,
        angular_correlation,
        angular_correlation_std,
        samples,
    ))

    #------------------#
    print("\tSaving correlation table to file '{:s}' ... ".format(args.output), end="")
    header = " ".join((
        "distance_in_unit_cells",
        "vector_dot_correlation",
        "normalized_correlation",
        "normalized_correlation_std",
        "angular_correlation",
        "angular_correlation_std",
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

    #------------------#
    if args.directional_output is not None:
        directional_angular_correlation_frames = np.asarray(
            directional_angular_correlation_frames
        )
        directional_angular_correlation_mean = np.mean(
            directional_angular_correlation_frames, axis=0
        )
        directional_angular_correlation_std = np.std(
            directional_angular_correlation_frames,
            axis=0,
            ddof=1 if n_frames > 1 else 0,
        )
        directional_displacement = directional_displacement.reshape(-1, 3)
        directional_distance = np.linalg.norm(directional_displacement, axis=1)
        directional_samples = np.full(
            len(directional_displacement), len(reference_coordinates) * n_frames
        )
        directional_output = np.column_stack((
            directional_displacement,
            directional_distance,
            directional_angular_correlation_mean,
            directional_angular_correlation_std,
            directional_samples,
        ))

        print("\tSaving direction-resolved angular correlations to file '{:s}' ... ".format(
            args.directional_output
        ), end="")
        header = " ".join((
            "displacement_i",
            "displacement_j",
            "displacement_k",
            "distance_in_unit_cells",
            "angular_correlation",
            "angular_correlation_std",
            "n_samples",
        ))
        np.savetxt(
            args.directional_output,
            directional_output,
            fmt=["%d"] * 3 + [float_format] * 3 + ["%d"],
            header=header,
            comments="",
        )
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
