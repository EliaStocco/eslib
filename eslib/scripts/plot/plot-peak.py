#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from eslib.formatting import esfmt

#---------------------------------------#
description = "Detect and plot a spike with its width from 1D time series."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", type=str, metavar="\b", required=True, help="input txt file")
    parser.add_argument("-o", "--output", type=str, metavar="\b", required=True, help="output file for the plot")
    parser.add_argument("-r", "--rel_height", type=float, metavar="\b", help="rel_height (default: %(default)s)", default=0.99)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print(f"\tReading data from '{args.input}' ... ", end="")
    data = np.loadtxt(args.input)
    print("done")

    x = np.arange(len(data))

    # Find peaks
    peaks, properties = find_peaks(data, prominence=0.5)

    if len(peaks) == 0:
        print("\tNo spike found.")
        peak = None
    else:
        # Choose the most prominent peak
        prominences = properties["prominences"]
        peak_idx = np.argmax(prominences)
        peak = peaks[peak_idx]

        # Get width at half-prominence
        results_half = peak_widths(data, [peak], rel_height=args.rel_height)
        width = results_half[0][0]
        left = results_half[2][0]
        right = results_half[3][0]

        print(f"\tDetected peak at index {peak} with width ≈ {width:.2f} samples")

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, data, label="Time series", marker='.',alpha=0.1)

    if peak is not None:
        # ax.plot(peak, data[peak], "rx", label="Detected peak")
        ax.fill_betweenx([0, data[peak]], left, right, color='red', alpha=1, label="Width @ Half Prominence")

    ax.set_xlabel("Index")
    ax.set_ylabel("Signal")
    ax.set_title("Spike Detection with Width")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    print(f"\tSaving plot to '{args.output}' ... ", end="")
    plt.savefig(args.output)
    print("done")

if __name__ == "__main__":
    main()
