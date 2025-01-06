#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from eslib.formatting import esfmt
from eslib.io_tools import pattern2sorted_files
from eslib.mathematics import mean_std_err

#---------------------------------------#
description = "Preprocess the data of the phonon projections and compute the average over the trajectories."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input" , **argv, required=True, type=str, help="input files")
    parser.add_argument("-dt", "--time_step" , **argv, required=True, type=float, help="time step in fs")
    parser.add_argument("-o", "--output", **argv, required=True, type=str, help="output file")
    return parser

@esfmt(prepare_args,description)
def main(args):

    print("\tSearching for files ... ", end="")
    files = pattern2sorted_files(args.input)
    print("done")
    print("\tn of files: ",len(files))
    Nmodes = len(files) 
    
    #------------------#
    y_max = -np.inf
    for file in files:
        data = pd.read_csv(file)
        y_max = max(y_max,data["mean"].max())
        
    #------------------#
       
    time = None
    
    # Use PdfPages to write multiple pages
    with PdfPages(args.output) as pdf:
        # Generate multiple plots
        for i in range(Nmodes):  # For example, creating 5 pages
            # Create a new figure
            
            data = pd.read_csv(files[i])
            if time is None:
                time = np.arange(data.shape[0])*args.time_step/1000 # fs tp ps
            
            plt.figure(figsize=(4,3))
            
            # Plot the data
            plt.plot(time,data["mean"])
            plt.fill_between(time,data["mean"]-data["err"],data["mean"]+data["err"],alpha=0.5,color="blue")
            plt.title("Mode {:d}".format(i))
            plt.ylim(0,y_max)
            
            # Add the current figure to the PDF
            pdf.savefig()  # Save the current figure into the PDF
            plt.close()    # Close the figure to free memory

        # # Optionally, add metadata to the PDF
        # pdf.infodict().update({
        #     'Title': 'Multi-Page PDF with Plots',
        #     'Author': 'Your Name',
        #     'Subject': 'Demonstrating multi-page PDF creation',
        #     'Keywords': 'Matplotlib, PDF, Multi-page',
        # })

    print(f"PDF file created: {args.output}")

if __name__ == "__main__":
    main()