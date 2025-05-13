#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from eslib.formatting import esfmt, float_format
from eslib.mathematics import pandas2ndarray, melt

#---------------------------------------#
description = "Autocorrelate unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--indices"     , **argv, required=True , type=str  , help="input file with the indices produced by 'divide-into-unitcells-with-indices.py'")
    parser.add_argument("-d" , "--dipoles"     , **argv, required=True , type=str  , help="*.txt file with the results produced by 'unit-cell-dipole.py', 'difference-unit-cell-dipoles.py', or 'analyse-unit-cell-dipoles.py'")
    parser.add_argument("-o" , "--output"      , **argv, required=True , type=str  , help="output file")
    return parser

#---------------------------------------#
def spatial_autocorrelation(x: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute circular autocorrelation along a specific axis using FFT.

    Args:
        x (np.ndarray): N-dimensional array.
        axis (int): Axis along which to compute circular autocorrelation.

    Returns:
        np.ndarray: Autocorrelation array with the same shape as input.
    """
    # Move the target axis to the first position
    #x = np.moveaxis(x, axis, 0)  # shape: (target_len, ...)
    
    # Compute FFT and power spectrum
    f = np.fft.fftn(x, axes=axis)
    power = f * np.conj(f)

    # Inverse FFT to get autocorrelation
    acorr = np.fft.ifftn(power, axes=axis)
    
    assert np.allclose(acorr.imag,0), "Imaginary part should be zero"
    acorr = acorr.real
    
    # normalize
    norm = np.prod([x.shape[i] for i in axis])
    acorr /= norm  # normalize by number of elements

    # Normalize
    m2 = np.mean(np.square(x),axis=axis,keepdims=True)
    acorr /= m2    

    # Move axis back to original position
    # acorr = np.moveaxis(acorr, 0, axis)

    return acorr

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
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
    reverse_indices,_ = pandas2ndarray(indices.reset_index(),index_columns=["L1","L2","L3"])
    reverse_indices = reverse_indices[:,:,:,0]
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
    print("\tComputing spatial correlation function ... ",end="")
    data, info = pandas2ndarray(dipoles,["structure","L1","L2","L3"],["unit_cell"])
    for n,k in enumerate(["L1","L2","L3"]):
        assert np.allclose(np.sort(info[k]),info[k]), f"indices of '{k}' not sorted."
            
    data -= np.mean(data,axis=(1,2,3),keepdims=True)
    autocorr =  spatial_autocorrelation(data,axis=(1,2,3))
    autocorr = np.linalg.norm(autocorr,axis=4)/np.sqrt(3.)
    
    result = melt(autocorr,index={0:"structure",1:"L1",2:"L2",3:"L3"},value_names=["autocorr"])
    print("done")
    
    #------------------#
    print("\tReading indices again from '{:s}' ... ".format(args.indices), end="")
    indices = np.loadtxt(args.indices).astype(int)
    indices = pd.DataFrame(data=indices,columns=["unit_cell", "L1", "L2","L3"])
    indices = indices.set_index(["L1", "L2","L3"])
    indices:pd.DataFrame = indices[~indices.index.duplicated(keep="first")]
    print("done")
    
    print("\tExtracting unit-cell indices ... ",end="")
    jj = result[["L1", "L2","L3"]].values
    result["unit_cell"] = reverse_indices[tuple(np.array(jj).T)]
    result = result[["structure","unit_cell", "L1", "L2","L3","autocorr"]]
    print("done")
    
    #------------------# 
    print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
    header = ''.join(f"{col:>15s}" for col in ["structure","unit_cell", "L1", "L2","L3"]) + ''.join(f"{col:>15s}" for col in ["autocorr"]) 
    np.savetxt(args.output,
        result.to_numpy(),
        fmt=["%15d"]*5 + [float_format],
        header=header,comments="")
    print("done")
        
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()
