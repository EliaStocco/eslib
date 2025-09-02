#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import linregress
from eslib.formatting import esfmt
from eslib.input import slist, flist
from eslib.plot import legend_options

#---------------------------------------#
# Description of the script's purpose
description = "Compute the peak position and width from some spectra."

# Expected input file format:
# Col 1: frequency in inversecm
# Col 2: infrared absorption spectrum in inversecm
# Col 3: std (over trajectories) of the spectrum
# Col 4: error of the spectrum (std/sqrt(N-1), with N = n. of trajectories)

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # input
    parser.add_argument("-i"  , "--input"     , **argv, required=True , type=slist, help="txt input files")
    parser.add_argument("-v"  , "--freq_range", **argv, required=False, type=flist, help="frequency range (default: %(default)s)", default=[0,np.inf])
    parser.add_argument("-x"  , "--x_values"  , **argv, required=True , type=flist, help="x values")
    parser.add_argument("-o" , "--output"     , **argv, required=False, type=str  , help="csv output file (default: %(default)s)", default="peaks.csv")
    parser.add_argument("-p" , "--plot"       , **argv, required=False, type=str  , help="plot file (default: %(default)s)", default="peaks.png")
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    data = [None]*len(args.input)
    for n,file in enumerate(args.input):
        print(f"\tReading file: {file} ... ",end="")
        data[n] = np.loadtxt(file,usecols=[0,1])
        print("done | shape: ",data[n].shape)
        
    #------------------#
    print("\n\tFinding peaks:")
    results = pd.DataFrame(columns=["file","x_value","peak_freq","peak_intensity","width"])
    
    for n, data_n in enumerate(data):
        
        freq = data_n[:,0]
        intensity = data_n[:,1]
        
        # frequency step
        dnu = np.diff(freq).mean()
        
        # Restrict to frequency range if specified
        mask = (freq >= args.freq_range[0]) & (freq <= args.freq_range[1])
        freq_masked = freq[mask]
        intensity_masked = intensity[mask]
        
        # Find peak
        
        peaks, _ = find_peaks(intensity_masked)
        peaks_freq = freq_masked[peaks]
        assert len(peaks) > 0, f"{args.x_values[n]}) no peaks found"
        if len(peaks) > 1:
            print(f"\t - {args.x_values[n]}) found multiple peaks at {peaks_freq}, taking the mean")
        
        mean_freq = np.mean(peaks_freq)
        peak = np.abs(freq_masked - mean_freq).argmin()
        
        if len(peaks) == 1:
            assert peak == peaks[0], f"{args.x_values[n]}) something went wrong in peak finding"
        
        peak_freq = freq_masked[peak]
        peak_intensity = intensity_masked[peak]
        width = peak_widths(intensity_masked, [peak], rel_height=0.8)[0][0] * dnu
        
        print(f"\t - {args.x_values[n]}) peak at {peak_freq:.2f} cm⁻¹, width ~ {width:.2f}")
        
        # Fill results
        results.loc[n] = {
            "file": args.input[n],
            "x_value": args.x_values[n],
            "peak_freq": peak_freq,
            "peak_intensity": peak_intensity,
            "width": width
        }
    
    #------------------#
    # Save results to CSV
    results = results.sort_values(by="x_value")
    print(f"\n\tSaving results to {args.output} ... ",end="")
    results.to_csv(args.output, index=False)
    print("done")
    
    #---------------------------------------#
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    quantities = ["peak_freq", "peak_intensity", "width"]
    titles = ["Peak Frequency (cm⁻¹)", "Peak Intensity", "Peak Width (cm⁻¹)"]

    slopes = {}

    for ax, quantity, title in zip(axs, quantities, titles):
        y = results[quantity].values
        x = results["x_value"].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        slopes[quantity] = slope
        
        # Plot data points
        ax.scatter(x, y, label="Data",color="blue")
        ax.plot(x, y, linestyle="solid",color="blue",alpha=0.5)
        
        # Plot regression line in red
        y_fit = intercept + slope * x
        ax.plot(x, y_fit, label=f"Slope = {slope:.4f}",linestyle="dashed",color="red")
        
        ax.set_ylabel(title)
        ax.legend(**legend_options)
        ax.grid(True)

    axs[-1].set_xlabel("X Value")
    plt.tight_layout()
    print(f"\n\tSaving plot to {args.plot} ... ",end="")
    plt.savefig(args.plot,dpi=300)
    print("done")
        
    return 0


#---------------------------------------#
if __name__ == "__main__":
    main()