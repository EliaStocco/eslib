#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from eslib.formatting import esfmt
from eslib.plot import generate_colors

#---------------------------------------#
description = "Plot one or more spatial correlations from a perovskite correlation table."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input", **argv, required=True, type=str, help="correlation table")
    parser.add_argument("-k", "--keywords", **argv, required=True, nargs="+", help="correlation columns to plot")
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output figure (default: %(default)s)", default="perovskite-correlation.png")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading correlation table from file '{:s}' ... ".format(args.input), end="")
    data = np.genfromtxt(args.input, names=True, dtype=float, encoding=None)
    print("done")
    if data.size == 0:
        raise ValueError("The correlation table is empty")
    if "distance_in_unit_cells" not in data.dtype.names:
        raise ValueError("The correlation table must contain 'distance_in_unit_cells'")

    missing = [key for key in args.keywords if key not in data.dtype.names]
    if missing:
        raise ValueError(
            "Unknown correlation columns: " + ", ".join(missing) + ". "
            "Available columns: " + ", ".join(data.dtype.names)
        )

    #------------------#
    print("\tPlotting correlations ... ", end="")
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = generate_colors(len(args.keywords), "tab10")
    distance = np.atleast_1d(data["distance_in_unit_cells"])
    for color, key in zip(colors, args.keywords):
        value = np.atleast_1d(data[key])
        axis.plot(distance, value, marker="o", markersize=3, color=color, label=key)
        std_key = f"{key}_std"
        if std_key in data.dtype.names:
            standard_deviation = np.atleast_1d(data[std_key])
            axis.fill_between(
                distance,
                value - standard_deviation,
                value + standard_deviation,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("Distance [unit cells]")
    axis.set_ylabel("Spatial correlation")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
