#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import iread, write
from eslib.formatting import esfmt
from eslib.functions import unique_elements

#---------------------------------------#
description = "Compute the A-B vector of every aggregate in a perovskite trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="input trajectory")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=str, help="frames to process (default: %(default)s)", default=":")
    parser.add_argument("-g" , "--aggregation" , **argv, required=False, type=str, help="per-atom array identifying the aggregate (default: %(default)s)", default="unit-cell")
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output extxyz trajectory (default: %(default)s)", default="perovskite-AB-vector.extxyz")
    return parser

#---------------------------------------#
def get_aggregate_indices(atoms:Atoms, aggregation:str):
    """Return one A and one B atom index for each aggregate."""
    if aggregation not in atoms.arrays:
        raise ValueError(
            f"The input structure does not have '{aggregation}' in atoms.arrays. "
            f"Available arrays: {', '.join(atoms.arrays.keys())}"
        )

    labels = np.asarray(atoms.arrays[aggregation])
    if labels.ndim not in (1, 2) or labels.shape[0] != len(atoms):
        raise ValueError(
            f"'{aggregation}' must have shape (n_atoms,) or (n_atoms, n_labels); "
            f"got {labels.shape} for {len(atoms)} atoms"
        )

    if labels.ndim == 1:
        keys = labels.tolist()
    else:
        keys = [tuple(row.tolist()) for row in labels]
    _, _, indices = unique_elements(keys)

    symbols = np.asarray(atoms.get_chemical_symbols())
    non_oxygen = symbols[symbols != "O"]
    site_symbols, _, _ = unique_elements(non_oxygen)
    if len(site_symbols) != 2:
        raise ValueError(
            "Removing oxygen atoms must leave exactly two chemical species; "
            f"found {site_symbols}"
        )
    a_symbol, b_symbol = site_symbols

    a_indices = np.empty(len(indices), dtype=int)
    b_indices = np.empty(len(indices), dtype=int)
    aggregate_labels = np.empty((len(indices), *labels.shape[1:]), dtype=labels.dtype)

    for n, index in enumerate(indices):
        index = np.asarray(index, dtype=int)
        a = index[symbols[index] == a_symbol]
        b = index[symbols[index] == b_symbol]
        if len(a) != 1 or len(b) != 1:
            raise ValueError(
                f"Aggregate {keys[index[0]]!r} contains {len(a)} '{a_symbol}' atoms "
                f"and {len(b)} '{b_symbol}' atoms; exactly one of each is required"
            )
        a_indices[n] = a[0]
        b_indices[n] = b[0]
        aggregate_labels[n] = labels[index[0]]

    return labels, aggregate_labels, a_indices, b_indices, a_symbol, b_symbol

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading and processing atomic structures from file '{:s}' ...".format(args.input))
    output_written = False
    reference_labels = None
    a_indices = None
    b_indices = None
    aggregate_labels = None
    a_symbol = None
    b_symbol = None

    read_kwargs = {"index": args.index}
    if args.input_format is not None:
        read_kwargs["format"] = args.input_format

    for frame, atoms in enumerate(iread(args.input, **read_kwargs)):
        labels, current_aggregate_labels, current_a_indices, current_b_indices, \
            current_a_symbol, current_b_symbol = get_aggregate_indices(atoms, args.aggregation)

        if reference_labels is None:
            reference_labels = labels.copy()
            aggregate_labels = current_aggregate_labels
            a_indices = current_a_indices
            b_indices = current_b_indices
            a_symbol = current_a_symbol
            b_symbol = current_b_symbol
            print("\tFound {:d} aggregates: A = {:s}, B = {:s}.".format(
                len(a_indices), a_symbol, b_symbol
            ))
        else:
            if not np.array_equal(labels, reference_labels):
                raise ValueError(
                    f"The '{args.aggregation}' assignment changes in frame {frame}. "
                    "A collective-variable trajectory requires a fixed aggregation."
                )
            if (current_a_symbol, current_b_symbol) != (a_symbol, b_symbol):
                raise ValueError(
                    f"The A/B species change in frame {frame}: "
                    f"expected ({a_symbol}, {b_symbol}), found "
                    f"({current_a_symbol}, {current_b_symbol})"
                )

        a_positions = atoms.positions[a_indices]
        b_positions = atoms.positions[b_indices]
        vector, distance = find_mic(
            b_positions - a_positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

        output = Atoms(
            symbols=[a_symbol] * len(a_indices),
            positions=a_positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
            info=atoms.info.copy(),
        )
        output.new_array(args.aggregation, aggregate_labels)
        output.new_array("A-index", a_indices)
        output.new_array("B-index", b_indices)
        output.new_array("AB-vector", vector)
        output.new_array("AB-distance", distance)
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
