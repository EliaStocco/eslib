#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.formatting import esfmt, eslog, float_format

PARALLEL = False  # Enable parallel processing

#---------------------------------------#
# Description of the script's purpose
description = "Check that the molecular dipoles are correct"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="csv produced by 'aggregate-into-molecules.py'")
    # parser.add_argument("-c" , "--cells"        , **argv, required=True , type=str, help="file with cells")
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output file")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # data
    with eslog(f"Reading data fram file '{args.input}'"):
        df = pd.read_csv(args.input)
        
    del df["Rx"]
    del df["Ry"]
    del df["Rz"]
    
    # Assuming `df` is your DataFrame
    dipole_cols = ['MACE_atomic_dipoles_0', 'MACE_atomic_dipoles_1', 'MACE_atomic_dipoles_2']

    # Group by time and sum the dipoles over all molecules
    df = df.groupby('time')[dipole_cols].sum().reset_index()
    df = df.sort_values(by="time")
    
    data = df[dipole_cols]
    
    np.savetxt(args.output,data,fmt=float_format)
            
    return
            
#---------------------------------------#
if __name__ == "__main__":
    main()


