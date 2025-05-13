#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog, float_format
from eslib.mathematics import melt

#---------------------------------------#
# Description of the script's purpose
description = "Save the cellpars to a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="input file")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="*.txt output file")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading the first atomic structure from file '{args.input}'"):
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
        
    #------------------#
    with eslog("Extracting cells"):
        cells = structures.get_cells()
        
    #------------------#
    with eslog("Extracting cellpars"):
        cellpars = np.asarray([ c.cellpar() for c in cells ])
    
    #------------------# 
    print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
    header = "cellpars: [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]"
    np.savetxt(args.output,
        cellpars,
        fmt=[float_format]*6,
        header=header)
    print("done")
    
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
    
    
# # Automatically detect unique values for each column
# n_structures = data['structure'].nunique()  # Number of unique structures
# n_components = data['component'].nunique()  # Number of unique components
# n_cells = data['unit_cell'].nunique()      # Number of unique unit cells

# # Pivot the data to have 'structure', 'component' as index and 'unit_cell' as columns
# pivoted = data.pivot_table(index=['structure', 'component'], columns='unit_cell', values='dipole', aggfunc='sum')

# # Convert the pivoted DataFrame to a numpy ndarray
# dipole_array = pivoted.to_numpy()

# # Reshape to 3D: (n_structures, n_components, n_cells)
# dipole_array = dipole_array.reshape(n_structures, n_cells,n_components)