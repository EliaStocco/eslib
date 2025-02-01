#!/usr/bin/env python
from typing import List, Tuple

import numpy as np
from ase import Atoms
import pandas as pd
import matplotlib.pyplot as plt

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
from eslib.input import flist
from eslib.physics import lorentzian
from eslib.tools import convert
from eslib.plot import legend_options

#---------------------------------------#
# Description of the script's purpose
description = "Compute the harmonic IR spectrum along the specified direction."
GAMMA = 1

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True, type=str  , help="input file with the atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-s" , "--summary"     , **argv, required=True , type=str  , help="summary file produced by 'vibrational-summary.py'")
    parser.add_argument("-d" , "--direction"   , **argv, required=True , type=flist, help="direction in Miller indices")
    parser.add_argument("-p" , "--plot"        , **argv, required=False, type=str  , help="plot file with the IR spectrum (default: %(default)s)", default='IR.par-perp.pdf')
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output file with the IR spectrum (default: %(default)s)", default='IR.par-perp.csv')
    return parser
            
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # atomic structure
    print("\tReading the displaced atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: {:d}".format(atoms.get_global_number_of_atoms()))

    #------------------#
    # summary
    print("\tReading the summary from file '{:s}' ... ".format(args.summary), end="")
    summary = pd.read_csv(args.summary)
    print("done")
    
    #------------------#
    # direction
    print("\tComputing the direction ... ", end="")
    cell = np.asarray(atoms.cell)
    direction = cell[0]*args.direction[0] + cell[1]*args.direction[1] + cell[2]*args.direction[2]
    direction:np.ndarray = direction / np.linalg.norm(direction)
    print("done")
    print("\t - direction: ",direction.tolist())
    
    #------------------#
    # Infrared spectrum
    # direction = np.asarray([1,0,0])
    Z = np.asarray(summary[ ["Z*x","Z*y","Z*z"]])
    IR_par = Z @ direction # IR spectrum parallel to the direction
    IR_tot = np.linalg.norm(Z,axis=1)
    IR_perp = np.sqrt( np.abs(IR_tot**2 - IR_par**2) ) # Pythagoras' theorem
    
    #------------------#
    # dataframe
    df = pd.DataFrame(columns=["index","w [THz]", "IR-par", "IR-perp", "IR-tot","IR-par %", "IR-perp %"])
    df["index"] = summary["index"]
    df["w [THz]"] = summary["w [THz]"]
    df["IR-par"] = IR_par
    df["IR-perp"] = IR_perp
    df["IR-tot"] = IR_tot
    df["IR-par %"] = IR_par / IR_tot * 100
    df["IR-perp %"] = IR_perp / IR_tot * 100
    
    print("\tWriting the IR spectrum to file '{:s}' ... ".format(args.output), end="")
    df.to_csv(args.output,index=False,float_format='%24.12f')
    print("done")
    
    #------------------#
    # plot
    print("\tPlotting the IR spectrum to file '{:s}' ... ".format(args.plot), end="")
    # Generate a range of x-values for the spectrum
    
    x_values = np.asarray(df["w [THz]"])
    x_plot = np.linspace(0, np.nanmax(x_values)*1.1, 100000)

    # Convolve each delta peak with the Lorentzian
    spectra = [np.zeros_like(x_plot),np.zeros_like(x_plot),np.zeros_like(x_plot)]
    gamma = convert(GAMMA,"frequency","inversecm","thz")
    for n,y_values in enumerate([IR_par,IR_perp,IR_tot]):
        for x0, y0 in zip(x_values, y_values):
            if np.isnan(x0):
                continue
            spectra[n] += lorentzian(x_plot, x0, y0,gamma)

    spectra = np.asarray(spectra)# [:2,:]
    # for n in range(2):
    #     spectra[n] = spectra[n] / np.max(spectra[2])
    # del spectra[2]

    # Create the plot
    fig,ax = plt.subplots(figsize=(6, 4))
    labels = ["parallel","perpendicular","total"]
    colors = ["blue","red","black"]
    spectra[2] = -spectra[2]
    # spectra[0] = -spectra[0]
    for n,spectrum in enumerate(spectra):
        ax.plot(x_plot, spectrum,color=colors[n],label=labels[n],linewidth=0.4)
    # ax.plot(x_plot,np.sqrt(spectra[0]**2 + spectra[1]**2),color="purple",label="test",linewidth=0.4)
    
    # # Add frequency annotations to the total spectrum
    # for n,Y in enumerate([IR_par]):
    #     for x, y in zip(x_values, Y):
    #         if not np.isnan(x) and y>2:  # Skip NaN values
    #             ax.text(x, y, f'{x:.2f} THz', fontsize=6, color="black", ha='left', va='bottom')


    plt.legend(**legend_options)
    # plt.ylim(-1.1,1.1)
    plt.xlim(min(x_plot),max(x_plot))
    plt.xlabel("Frequency [THz]")
    plt.ylabel('Intensity [arb. unit]')
    plt.title('Harmonic IR spectrum')
    plt.grid()
    plt.tight_layout()
    plt.savefig(args.plot,bbox_inches='tight',dpi=300)
    print("done")
    

    pass
    
#---------------------------------------#
if __name__ == "__main__":
    main()