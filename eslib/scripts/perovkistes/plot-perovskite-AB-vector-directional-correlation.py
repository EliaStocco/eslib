#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from eslib.formatting import esfmt

#---------------------------------------#
description = "Plot the direction-resolved angular correlation of perovskite A-B vectors."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input", **argv, required=True, type=str, help="direction-resolved correlation table")
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output figure (default: %(default)s)", default="perovskite-AB-vector-directional-correlation.png")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading direction-resolved correlations from file '{:s}' ... ".format(args.input), end="")
    data = np.genfromtxt(args.input, names=True, dtype=float, encoding=None)
    print("done")
    if data.size == 0:
        raise ValueError("The correlation table is empty")

    required = (
        "displacement_i",
        "displacement_j",
        "displacement_k",
        "angular_correlation",
        "angular_correlation_std",
    )
    missing = [key for key in required if key not in data.dtype.names]
    if missing:
        raise ValueError(
            "The correlation table is missing: " + ", ".join(missing)
        )

    displacement = np.column_stack([np.atleast_1d(data[key]) for key in required[:3]])
    correlation = np.atleast_1d(data["angular_correlation"])
    standard_deviation = np.atleast_1d(data["angular_correlation_std"])

    #------------------#
    print("\tPlotting direction-resolved angular correlations ... ", end="")
    figure = plt.figure(figsize=(12, 5.5))
    axes = [
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    ]

    correlation_plot = axes[0].scatter(
        displacement[:, 0], displacement[:, 1], displacement[:, 2],
        c=correlation, cmap="coolwarm", vmin=-1, vmax=1, s=36,
    )
    standard_deviation_plot = axes[1].scatter(
        displacement[:, 0], displacement[:, 1], displacement[:, 2],
        c=standard_deviation, cmap="viridis", vmin=0,
        vmax=max(np.max(standard_deviation), np.finfo(float).eps), s=36,
    )
    figure.colorbar(correlation_plot, ax=axes[0], shrink=0.75, label=r"$C_{\mathrm{ang}}(\Delta\mathbf{n})$")
    figure.colorbar(standard_deviation_plot, ax=axes[1], shrink=0.75, label="Standard deviation")

    for axis, title in zip(axes, ("Angular correlation", "Frame-to-frame standard deviation")):
        axis.set_xlabel(r"$\Delta i$")
        axis.set_ylabel(r"$\Delta j$")
        axis.set_zlabel(r"$\Delta k$")
        axis.set_title(title)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=25, azim=40)

    figure.tight_layout()
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
