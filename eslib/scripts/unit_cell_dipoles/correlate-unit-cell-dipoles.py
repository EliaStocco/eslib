#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from eslib.formatting import esfmt, float_format
from eslib.mathematics import pandas2ndarray, mean_std_err

#---------------------------------------#
description = "Correlate unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--indices"     , **argv, required=True , type=str  , help="input file with the indices produced by 'divide-into-unitcells-with-indices.py'")
    parser.add_argument("-d" , "--dipoles"     , **argv, required=True , type=str  , help="*.txt file with the results produced by 'unit-cell-dipole.py', 'difference-unit-cell-dipoles.py', or 'analyse-unit-cell-dipoles.py'")
    parser.add_argument("-o" , "--output"      , **argv, required=True , type=str  , help="output folder")
    return parser

#---------------------------------------#
def circular_autocorrelation_along_axis(x: np.ndarray, axis: int, mean:np.ndarray=0.0) -> np.ndarray:
    """
    Compute circular autocorrelation along a specific axis using FFT.

    Args:
        x (np.ndarray): N-dimensional array.
        axis (int): Axis along which to compute circular autocorrelation.

    Returns:
        np.ndarray: Autocorrelation array with the same shape as input.
    """
    # Move the target axis to the first position
    x = np.moveaxis(x, axis, 0)  # shape: (target_len, ...)

    # Center the data
    # x_mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    # Compute FFT and power spectrum
    f = np.fft.fft(x_centered, axis=0)
    power = f * np.conj(f)

    # Inverse FFT to get autocorrelation
    acorr = np.fft.ifft(power, axis=0).real

    # Normalize
    var = np.var(x, axis=0, keepdims=True)
    acorr /= var
    acorr /= x.shape[0]  # normalize by number of elements

    # Move axis back to original position
    acorr = np.moveaxis(acorr, 0, axis)

    return acorr

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    os.makedirs(args.output, exist_ok=True)
    
    #------------------#
    print("\tReading dipoles from '{:s}' ... ".format(args.dipoles), end="")
    dipoles = pd.read_csv(args.dipoles, sep='\s+')
    dipoles["structure"] = dipoles["structure"].astype(int)
    dipoles["unit_cell"]  = dipoles["unit_cell"].astype(int)
    # dipole_array = dcast(dipoles, ["structure","unit_cell"])
    print("done")
    
    #------------------#
    print("\tReading indices from '{:s}' ... ".format(args.indices), end="")
    indices = np.loadtxt(args.indices).astype(int)
    indices = pd.DataFrame(data=indices,columns=["unit_cell", "L1", "L2","L3"])
    indices = indices.set_index("unit_cell")
    indices:pd.DataFrame = indices[~indices.index.duplicated(keep="first")]
    # c_indices = dcast(indices, ["unit_cell"])
    print("done")
    
    #------------------#
    a = np.sort(np.unique(dipoles["unit_cell"]))
    b = np.sort(np.unique(indices.index))
    assert np.allclose(a,b), f"coding error"
    
    #------------------#
    Lxyz = indices.loc[dipoles["unit_cell"].values] # np.asarray([ indices.loc[a] for a in dipoles["unit_cell"].values ])
    for n,k in enumerate(["L1","L2","L3"]):
        dipoles[k] = np.asarray(Lxyz[k])
        
    #------------------#
    dipoles["perpendicular"] = np.nan
    
    for n,k in enumerate(["L1","L2","L3"]):
        
        print(f"\n\tProcessing {k} ... ", end="")
        if k == "L1":
            perpendicular_directions = ["L2","L3"]
        elif k == "L2":
            perpendicular_directions = ["L1","L3"]
        elif k == "L3":
            perpendicular_directions = ["L1","L2"]
        perpendicular_unit_cells = dipoles[perpendicular_directions]
        # unique values, unique values indices
        uv,uvi = np.unique(perpendicular_unit_cells.values,axis=0,return_inverse=True)
        
        # tuples = list(map(tuple, perpendicular_unit_cells.values))
        # dipoles["perpendicular"] = tuples

        dipoles["perpendicular"] = uvi

        df  = dipoles.copy()
        # df["dipole"] = df[["dipole_x","dipole_y","dipole_z"]]
        for p in perpendicular_directions+["unit_cell"]:
            del df[p]    
            
        gdf, info = pandas2ndarray(df, ["structure","perpendicular",k])
        autocorr = circular_autocorrelation_along_axis(gdf, axis=2)
        autocorr = np.linalg.norm(autocorr,axis=3)
        autocorr = mean_std_err(autocorr, axis=1)
        
        print("done")
        
        common_header  = "\nrow: structure\ncolumn: |L| (distance)"
        
        ofile = f"{args.output}/{k}.mean.txt"
        print(f"\tSaving {k} mean to file '{ofile}' ", end="")
        np.savetxt(ofile, autocorr[0], fmt=float_format,header=f"{k} mean"+common_header)
        print("done")
        
        ofile = f"{args.output}/{k}.std.txt"
        print(f"\tSaving {k} std to file '{ofile}' ", end="")
        np.savetxt(ofile, autocorr[1], fmt=float_format,header=f"{k} std"+common_header)
        print("done")
        
        ofile = f"{args.output}/{k}.err.txt"
        print(f"\tSaving {k} err to file '{ofile}' ", end="")
        np.savetxt(ofile, autocorr[2], fmt=float_format,header=f"{k} err"+common_header)
        print("done")
        
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
