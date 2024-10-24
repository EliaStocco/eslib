#!/usr/bin/env python
import os
import sys
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np

from eslib.formatting import esfmt
from eslib.input import slist
from eslib.plot import vzero

#---------------------------------------#
description="Create a band plot as in phonopy"

#---------------------------------------#
def get_options(description):
    # Parse options
    import argparse
    argv = {"metavar" : "\b",}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i","--input"  , **argv, required=True , type=slist, help="*yaml input files")
    parser.add_argument("-l","--labels"  , **argv, required=True , type=slist, help="labels")
    parser.add_argument("-f1","--fmin"  , **argv, required=False, type=float, help="minimum frequency (default: %(default)s)", default=0)
    parser.add_argument("-f2","--fmax"  , **argv, required=False, type=float, help="maximum frequency (default: %(default)s)", default=32)    
    parser.add_argument("-xtl","--xticklabels", **argv, required=False, type=str, help="x tick labels (default: %(default)s)", default="[r'$\\Gamma$', 'T', r'$\\rm H_{2}$', r'$\\rm H_{0}$', 'L', r'$\\Gamma$', r'$\\rm S_{0}$', r'$\\rm S_{2}$', 'F', r'$\\Gamma$']")
    parser.add_argument("-o" ,"--output", **argv, required=False, type=str  , help="output file (default: %(default)s)", default="bandplot.pdf")
    return parser

#---------------------------------------#
def _get_label_for_latex(label):
    return label.replace("_", r"\_")


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

    args.xticklabels = literal_eval(args.xticklabels)
    
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

    # Plot band structures
    fmts = ["r-","b-","g-","c-","m-","y-","k-","r--","b--","g--","c--","m--","y--","k--"]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylim(args.fmin,args.fmax)
    xmin = +np.inf
    xmax = -np.inf
    vertical_lines = []
    for n, label in enumerate(args.labels):
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
    ax.set_ylabel("Frequency [THz]")
    plt.grid(axis="y")
    # plt.show()


    if args.output is not None:
        _savefig(plt, args.output)
    else:
        plt.show()

#---------------------------------------#
if __name__ == "__main__":
    main()
