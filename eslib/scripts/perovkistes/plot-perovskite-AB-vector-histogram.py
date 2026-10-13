#!/usr/bin/env python
import pathlib

import numpy as np
from ase.io import iread
from eslib.formatting import esfmt

#---------------------------------------#
description = "Plot the modulus and orientation distribution of perovskite A-B vectors."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str, help="extxyz produced by compute-perovskite-AB-vector.py")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"       , **argv, required=False, type=str, help="frames to process (default: %(default)s)", default=":")
    parser.add_argument("-bd", "--distance_bins", **argv, required=False, type=int, help="number of bins for the distance histogram (default: %(default)s)", default=100)
    parser.add_argument("-mp", "--max_points", **argv, required=False, type=int, help="maximum number of vectors in the orientation plot (default: %(default)s)", default=100000)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str, help="output HTML (interactive) or image/PDF (static) file (default: %(default)s)", default="perovskite-AB-vector-histogram.html")
    return parser

#---------------------------------------#
def get_vectors(atoms, frame:int):
    if "AB-vector" not in atoms.arrays:
        raise ValueError(
            "The input does not contain the 'AB-vector' array. It must be the "
            "output of compute-perovskite-AB-vector.py."
        )

    vectors = np.asarray(atoms.arrays["AB-vector"], dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(
            f"'AB-vector' in frame {frame} must have shape (n_aggregates, 3); "
            f"got {vectors.shape}"
        )
    return vectors

#---------------------------------------#
def plot_static(distance, unit_vectors, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 5))
    grid = figure.add_gridspec(1, 2, width_ratios=(1, 1.3), wspace=0.28)
    distance_axis = figure.add_subplot(grid[0])
    orientation_axis = figure.add_subplot(grid[1], projection="3d")

    distance_axis.hist(distance, bins=args.distance_bins, edgecolor="black", color="tab:blue")
    distance_axis.set_xlabel(r"$|\mathbf{r}_{AB}|$ [$\AA$]")
    distance_axis.set_ylabel("Count")
    distance_axis.set_title("A-B distance")
    distance_axis.grid(alpha=0.3)

    orientation_axis.scatter(
        unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2],
        s=1, alpha=0.1, color="tab:orange", depthshade=False,
    )
    orientation_axis.set_xlabel(r"$\hat{r}_{AB,x}$")
    orientation_axis.set_ylabel(r"$\hat{r}_{AB,y}$")
    orientation_axis.set_zlabel(r"$\hat{r}_{AB,z}$")
    orientation_axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    orientation_axis.set_box_aspect((1, 1, 1))
    orientation_axis.set_title("A-B vector orientation with wall projections")

    # XY, XZ, and YZ projections on the z=-1, y=+1, and x=-1 walls.
    wall_projections = (
        (unit_vectors[:, 0], unit_vectors[:, 1], np.full(len(unit_vectors), -1)),
        (unit_vectors[:, 0], np.full(len(unit_vectors), 1), unit_vectors[:, 2]),
        (np.full(len(unit_vectors), -1), unit_vectors[:, 1], unit_vectors[:, 2]),
    )
    for x, y, z in wall_projections:
        orientation_axis.scatter(
            x, y, z, s=1, alpha=0.07, color="tab:blue", depthshade=False,
        )

    crosshairs = (
        ((-1, 1), (0, 0), (-1, -1)), ((0, 0), (-1, 1), (-1, -1)),
        ((-1, 1), (1, 1), (0, 0)), ((0, 0), (1, 1), (-1, 1)),
        ((-1, -1), (-1, 1), (0, 0)), ((-1, -1), (0, 0), (-1, 1)),
    )
    for x, y, z in crosshairs:
        orientation_axis.plot(x, y, z, color="0.4", linewidth=0.7)

    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)

#---------------------------------------#
def plot_interactive(distance, unit_vectors, args):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "Interactive plotting requires plotly. Install eslib with its dependencies "
            "or run 'python -m pip install plotly'."
        ) from error

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        column_widths=(0.42, 0.58),
        subplot_titles=("A-B distance", "A-B vector orientation with wall projections"),
    )
    figure.add_trace(
        go.Histogram(
            x=distance,
            nbinsx=args.distance_bins,
            marker={"color": "#1f77b4", "line": {"color": "black", "width": 0.7}},
            name="A-B distance",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter3d(
            x=unit_vectors[:, 0],
            y=unit_vectors[:, 1],
            z=unit_vectors[:, 2],
            mode="markers",
            marker={"size": 2, "color": "#ff7f0e", "opacity": 0.25},
            name="orientation",
            hovertemplate=(
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>orientation</extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # XY, XZ, and YZ projections on the z=-1, y=+1, and x=-1 walls.
    wall_projections = (
        (unit_vectors[:, 0], unit_vectors[:, 1], np.full(len(unit_vectors), -1), "xy projection"),
        (unit_vectors[:, 0], np.full(len(unit_vectors), 1), unit_vectors[:, 2], "xz projection"),
        (np.full(len(unit_vectors), -1), unit_vectors[:, 1], unit_vectors[:, 2], "yz projection"),
    )
    for x, y, z, name in wall_projections:
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={"size": 1.5, "color": "#1f77b4", "opacity": 0.18},
                name=name,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    crosshairs = (
        ((-1, 1), (0, 0), (-1, -1)), ((0, 0), (-1, 1), (-1, -1)),
        ((-1, 1), (1, 1), (0, 0)), ((0, 0), (1, 1), (-1, 1)),
        ((-1, -1), (-1, 1), (0, 0)), ((-1, -1), (0, 0), (-1, 1)),
    )
    for x, y, z in crosshairs:
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "#666666", "width": 3},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    figure.update_xaxes(title_text=r"|r<sub>AB</sub>| [Å]", row=1, col=1)
    figure.update_yaxes(title_text="Count", row=1, col=1)
    figure.update_layout(
        width=1300,
        height=650,
        bargap=0.02,
        template="plotly_white",
        legend={"x": 0.99, "xanchor": "right", "y": 0.99},
        scene={
            "xaxis": {"title": "r̂<sub>AB,x</sub>", "range": [-1, 1]},
            "yaxis": {"title": "r̂<sub>AB,y</sub>", "range": [-1, 1]},
            "zaxis": {"title": "r̂<sub>AB,z</sub>", "range": [-1, 1]},
            "aspectmode": "cube",
        },
    )
    figure.write_html(args.output, include_plotlyjs=True)

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.distance_bins <= 0 or args.max_points <= 0:
        raise ValueError("--distance_bins and --max_points must be positive")
    interactive = pathlib.Path(args.output).suffix.lower() in (".html", ".htm")

    read_kwargs = {"index": args.index}
    if args.input_format is not None:
        read_kwargs["format"] = args.input_format

    #------------------#
    print("\tReading A-B vectors from file '{:s}' ...".format(args.input))
    vectors = []
    for frame, atoms in enumerate(iread(args.input, **read_kwargs)):
        vectors.append(get_vectors(atoms, frame))
        if (frame + 1) % 100 == 0:
            print("\tRead {:d} frames.".format(frame + 1))

    if len(vectors) == 0:
        raise ValueError("No structures were selected from the input trajectory")

    vectors = np.concatenate(vectors, axis=0)
    distance = np.linalg.norm(vectors, axis=1)
    nonzero = distance > np.finfo(float).eps
    if not np.any(nonzero):
        raise ValueError("All A-B vectors have zero length, so their orientation is undefined")

    unit_vectors = vectors[nonzero] / distance[nonzero, np.newaxis]
    if len(unit_vectors) > args.max_points:
        random = np.random.default_rng(seed=0)
        selected = random.choice(len(unit_vectors), size=args.max_points, replace=False)
        unit_vectors = unit_vectors[selected]
        print("\tUsing {:d} vectors for the orientation plot.".format(len(unit_vectors)))

    #------------------#
    print("\tPlotting {:d} A-B vectors from {:d} frames ...".format(len(distance), frame + 1))
    output_type = "interactive" if interactive else "static"
    print("\tSaving {:s} figure to file '{:s}' ... ".format(output_type, args.output), end="")
    if interactive:
        plot_interactive(distance, unit_vectors, args)
    else:
        plot_static(distance, unit_vectors, args)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
