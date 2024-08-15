#!/usr/bin/env python
import os
import sys
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import gzip
import sys
import warnings
from typing import Union
import numpy as np
import yaml
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.units import VaspToTHz
from eslib.formatting import esfmt
from eslib.input import slist
from eslib.plot import vzero
# matplotlib.use("Agg")

#---------------------------------------#
description="Phonopy bandplot command-line-tool"

#---------------------------------------#
def get_options(description):
    # Parse options
    import argparse
    argv = {"metavar" : "\b",}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i","--input"  , **argv, required=True , type=slist, help="*yaml input files")
    parser.add_argument("-f1","--fmin"  , **argv, required=False, type=float, help="minimum frequency (default: %(default)s)", default=0)
    parser.add_argument("-f2","--fmax"  , **argv, required=False, type=float, help="maximum frequency (default: %(default)s)", default=32)    
    parser.add_argument("-xtl","--xticklabels", **argv, required=False, type=slist, help="x marks (default: %(default)s)", default=['$\\Gamma$', 'T', '${\\rm H}_2$','${\\rm H}_0$', 'L', '$\\Gamma$', '${\\rm S}_0$','${\\rm S}_2$', 'F', '$\\Gamma$'])
    parser.add_argument("-o" ,"--output", **argv, required=False, type=str  , help="output file (default: %(default)s)", default="bandplot.pdf")
    return parser

#---------------------------------------#
def estimate_band_connection(prev_eigvecs, eigvecs, prev_band_order):
    """Connect neighboring qpoints by eigenvector similarity."""
    metric = np.abs(np.dot(prev_eigvecs.conjugate().T, eigvecs))
    connection_order = []
    for overlaps in metric:
        maxval = 0
        for i in reversed(range(len(metric))):
            val = overlaps[i]
            if i in connection_order:
                continue
            if val > maxval:
                maxval = val
                maxindex = i
        connection_order.append(maxindex)

    band_order = [connection_order[x] for x in prev_band_order]

    return band_order


#---------------------------------------#
def get_band_qpoints_and_path_connections(band_paths, npoints=51, rec_lattice=None):
    """Return qpoints and connections of paths."""
    path_connections = []
    for paths in band_paths:
        path_connections += [
            True,
        ] * (len(paths) - 2)
        path_connections.append(False)
    return (
        get_band_qpoints(band_paths, npoints=npoints, rec_lattice=rec_lattice),
        path_connections,
    )


def get_band_qpoints(band_paths, npoints=51, rec_lattice=None):
    """Generate qpoints for band structure path.

    Note
    ----

    Behavior changes with and without rec_lattice given.

    Parameters
    ----------
    band_paths: list of array_likes
        Sets of end points of paths
        dtype='double'
        shape=(sets of paths, paths, 3)

        example:
            [[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]],
             [[0.5, 0.25, 0.75], [0, 0, 0]]]

    npoints: int, optional
        Number of q-points in each path including end points. Default is 51.

    rec_lattice: array_like, optional
        When given, q-points are sampled in a similar interval. The longest
        path length divided by npoints including end points is used as the
        reference interval. Reciprocal basis vectors given in column vectors.
        dtype='double'
        shape=(3, 3)

    """
    npts = _get_npts(band_paths, npoints, rec_lattice)
    qpoints_of_paths = []
    c = 0
    for band_path in band_paths:
        nd = len(band_path)
        for i in range(nd - 1):
            delta = np.subtract(band_path[i + 1], band_path[i]) / (npts[c] - 1)
            qpoints = [delta * j for j in range(npts[c])]
            qpoints_of_paths.append(np.array(qpoints) + band_path[i])
            c += 1

    return qpoints_of_paths


def get_band_qpoints_by_seekpath(primitive, npoints, is_const_interval=False):
    """q-points along BZ high symmetry paths are generated using seekpath.

    Parameters
    ----------
    primitive : PhonopyAtoms
        Primitive cell.
    npoints : int
        Number of q-points sampled along a path including end points.
    is_const_interval : bool, optional
        When True, q-points are sampled in a similar interval. The longest
        path length divided by npoints including end points is used as the
        reference interval. Default is False.

    Returns
    -------
    bands : List of ndarray
        Sets of qpoints that can be passed to phonopy.set_band_structure().
        shape of each ndarray : (npoints, 3)
    labels : List of pairs of str
        Symbols of end points of paths.
    connections : List of bool
        This gives one path is connected to the next path, i.e., if False,
        there is a jump of q-points. Number of elements is the same at
        that of paths.

    """
    try:
        import seekpath
    except ImportError:
        raise ImportError("You need to install seekpath.")

    band_path = seekpath.get_path(primitive.totuple())
    point_coords = band_path["point_coords"]
    qpoints_of_paths = []
    if is_const_interval:
        reclat = np.linalg.inv(primitive.cell)
    else:
        reclat = None
    band_paths = [
        [point_coords[path[0]], point_coords[path[1]]] for path in band_path["path"]
    ]
    npts = _get_npts(band_paths, npoints, reclat)
    for c, path in enumerate(band_path["path"]):
        q_s = np.array(point_coords[path[0]])
        q_e = np.array(point_coords[path[1]])
        band = [q_s + (q_e - q_s) / (npts[c] - 1) * i for i in range(npts[c])]
        qpoints_of_paths.append(band)
    labels, path_connections = _get_labels(band_path["path"])

    return qpoints_of_paths, labels, path_connections


def band_plot(axs, frequencies, distances, path_connections, labels, fmt="r-"):
    """Return band structure plot."""
    bp = BandPlot(axs)
    bp.decorate(labels, path_connections, frequencies, distances)
    bp.plot(distances, frequencies, path_connections, fmt=fmt)


def _get_npts(band_paths, npoints, rec_lattice):
    """Return numbers of qpoints of band segments."""
    if rec_lattice is not None:
        path_lengths = []
        for band_path in band_paths:
            nd = len(band_path)
            for i in range(nd - 1):
                vector = np.subtract(band_path[i + 1], band_path[i])
                length = np.linalg.norm(np.dot(rec_lattice, vector))
                path_lengths.append(length)
        max_length = max(path_lengths)
        npts = [np.rint(pl / max_length * npoints).astype(int) for pl in path_lengths]
    else:
        npts = [
            npoints,
        ] * np.sum([len(paths) for paths in band_paths])

    for i, npt in enumerate(npts):
        if npt < 2:
            npts[i] = 2

    return npts


def _get_labels(pairs_of_symbols):
    path_connections = []
    labels = []

    for i, pairs in enumerate(pairs_of_symbols[:-1]):
        if pairs[1] != pairs_of_symbols[i + 1][0]:
            path_connections.append(False)
            labels += list(pairs)
        else:
            path_connections.append(True)
            labels.append(pairs[0])
    path_connections.append(False)
    labels += list(pairs_of_symbols[-1])

    for i, l in enumerate(labels):
        if "GAMMA" in l:
            labels[i] = "$" + l.replace("GAMMA", r"\Gamma") + "$"
        elif "SIGMA" in l:
            labels[i] = "$" + l.replace("SIGMA", r"\Sigma") + "$"
        elif "DELTA" in l:
            labels[i] = "$" + l.replace("DELTA", r"\Delta") + "$"
        elif "LAMBDA" in l:
            labels[i] = "$" + l.replace("LAMBDA", r"\Lambda") + "$"
        else:
            labels[i] = r"$\mathrm{%s}$" % l

    return labels, path_connections


def _get_label_for_latex(label):
    return label.replace("_", r"\_")


def _get_max_frequency(frequencies):
    return max([np.max(fq) for fq in frequencies])


def _find_wrong_path_connections(all_path_connections):
    for i, path_connections in enumerate(all_path_connections):
        if path_connections != all_path_connections[0]:
            return i
    return 0


def _arrange_band_data(distances, frequencies, qpoints, segment_nqpoints, label_pairs):
    i = 0
    freq_list = []
    dist_list = []
    qpt_list = []
    
    for nq in segment_nqpoints:
        freq_list.append(frequencies[i : (i + nq)])
        dist_list.append(distances[i : (i + nq)])
        qpt_list.append(qpoints[i : (i + nq)])
        i += nq

    qpath = [qpt_list[0][0]]
    for i, qpts in enumerate(qpt_list):
        qpath.append(qpts[-1])
    qpath = np.asarray(qpath)

    if not label_pairs:
        labels = None
        path_connections = []
        if len(qpt_list) > 1:
            for i, qpts in enumerate(qpt_list[1:]):
                if (np.abs(qpt_list[i][-1] - qpts[0]) < 1e-5).all():
                    path_connections.append(True)
                else:
                    path_connections.append(False)
        path_connections += [
            False,
        ]
    else:
        labels = []
        path_connections = []
        if len(label_pairs) > 1:
            for i, pair in enumerate(label_pairs[1:]):
                labels.append(label_pairs[i][0])
                if label_pairs[i][1] != pair[0]:
                    labels.append(label_pairs[i][1])
                    path_connections.append(False)
                else:
                    path_connections.append(True)
            if label_pairs[-2][1] != label_pairs[-1][1]:
                labels += label_pairs[-1]
            else:
                labels.append(label_pairs[-1][1])
        else:
            labels += label_pairs[0]
        path_connections += [
            False,
        ]

    return labels, path_connections, freq_list, dist_list, qpath


def _savefig(plt, file, fonttype=42, family="serif"):
    plt.tight_layout()
    plt.rcParams["pdf.fonttype"] = fonttype
    plt.rcParams["font.family"] = family
    plt.savefig(file)

def _read_band_yaml(filename):

    try:
        import yaml
    except ImportError:
        print("You need to install python-yaml.")
        sys.exit(1)

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    _, ext = os.path.splitext(filename)
    if ext == ".xz" or ext == ".lzma":
        try:
            import lzma
        except ImportError:
            raise (
                "Reading a lzma compressed file is not supported "
                "by this python version."
            )
        with lzma.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    elif ext == ".gz":
        import gzip

        with gzip.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    else:
        with open(filename, "r") as f:
            data = yaml.load(f, Loader=Loader)

    frequencies = []
    distances = []
    qpoints = []
    labels = []
    for j, v in enumerate(data["phonon"]):
        if "label" in v:
            labels.append(v["label"])
        else:
            labels.append(None)
        frequencies.append([f["frequency"] for f in v["band"]])
        qpoints.append(v["q-position"])
        distances.append(v["distance"])

    if "labels" in data:
        labels = data["labels"]
    elif all(x is None for x in labels):
        labels = []

    return (
        np.array(distances),
        np.array(frequencies),
        np.array(qpoints),
        data["segment_nqpoint"],
        labels,
    )

@esfmt(get_options,description)
def main(args):

    bands_data = [None]*len(args.input)
    print("\tReading the input files:")
    for n,file in enumerate(args.input):
        print("\t - {:d}: '{:s}' ... ".format(n,file), end="")
        bands_data[n] = _read_band_yaml(file)
        print("done")

    print("\n\tPreprocessing data:")
    plots_data = [None]*len(args.input)
    for n,file in enumerate(args.input):
        print("\t - {:d}: '{:s}' ... ".format(n,file), end="")
        plots_data[n] = _arrange_band_data(*bands_data[n])
        print("done")


    print("\n\tCheck consistency:", end="")
    # Check consistency of input band structures
    all_path_connections = [data[1] for data in plots_data]
    wrong_file_i = _find_wrong_path_connections(all_path_connections)
    if wrong_file_i > 0:
        raise RuntimeError(
            "Band path of %s is inconsistent with %s."
            % (args.input[wrong_file_i], args.input[0])
        )

    # Decoration of figure
    # max_frequencies = [_get_max_frequency(data[2]) for data in plots_data]
    # plot_data = plots_data[np.argmax(max_frequencies)]
    # _, path_connections, _, _ = plot_data
    # n = len([x for x in path_connections if not x])
    # fig, ax = plt.subplots(figsize=(5,5))
    # axs = ImageGrid(
    #     fig,
    #     111,  # similar to subplot(111)
    #     nrows_ncols=(1, n),
    #     axes_pad=0.11,
    #     label_mode="L",
    # )
    # for ax in axs:

    # band_plot = BandPlot([ax])
    # band_plot.set_xscale_from_data(plot_data[2], plot_data[3])
    # band_plot.xscale = band_plot.xscale * args.factor
    # band_plot.decorate(*plot_data)

    # Plot band structures
    fmts = [
        "r-",
        "b-",
        "g-",
        "c-",
        "m-",
        "y-",
        "k-",
        "r--",
        "b--",
        "g--",
        "c--",
        "m--",
        "y--",
        "k--",
    ]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylim(args.fmin,args.fmax)
    xmin = +np.inf
    xmax = -np.inf
    vertical_lines = []
    for n, label in enumerate(args.input):
        _, path_connections, frequencies, distances, qpath = plots_data[n]
        fmt = fmts[n % len(fmts)]
        # _f = [f_seg  for f_seg in frequencies]
        # if args.show_legend:
        label=_get_label_for_latex(label)
        # band_plot.plot(d, _f, p, fmt=fmt, label=_get_label_for_latex(label))
        # else:
        #     band_plot.plot(d, _f, p, fmt=fmt)
        if fmt is None:
            _fmt = "r-"
        else:
            _fmt = fmt

        count = 0
        # distances_scaled = [d for d in distances]
        for i, (d, f, c) in enumerate(
            zip(distances, frequencies, path_connections)
        ):
            # ax = self._axs[count]
            if i == 0 and label is not None:
                curves = ax.plot(d, f, _fmt, linewidth=1)
                curves[0].set_label(label)
                ax.legend(facecolor='white', framealpha=1,edgecolor="black",loc="upper right")
            else:
                ax.plot(d, f, _fmt, linewidth=1)
            
            if n == 0 :
                vertical_lines.append(d[0])
                # vzero(ax,d[0])
            xmin = min(xmin, min(d))
            xmax = max(xmax, max(d))
            if not c:
                count += 1

    vertical_lines = np.asarray(vertical_lines+[xmax])
    vertical_lines = np.unique(vertical_lines)
    # plt.show()
    for v in vertical_lines:
        vzero(ax,v,linestyle="solid",linewidth=0.5)

    if args.xticklabels is not None:
        ax.set_xticks(vertical_lines)
        ax.set_xticklabels(args.xticklabels)

    
    ax.set_xlim(0,xmax)
    plt.grid(axis="y")
    # plt.show()


    if args.output is not None:
        _savefig(plt, args.output)
    else:
        plt.show()

#---------------------------------------#
if __name__ == "__main__":
    main()
