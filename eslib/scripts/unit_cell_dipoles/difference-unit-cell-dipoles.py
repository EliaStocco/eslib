#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.formatting import esfmt, float_format 

#---------------------------------------#
# Description of the script's purpose
description = "Plot the unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"    , **argv, required=True, type=str, help="*.txt file with the results produced by 'unit-cell-dipole.py'")
    parser.add_argument("-r" , "--reference", **argv, required=True, type=str, help="*.txt file with the reference results")
    parser.add_argument("-o" , "--output"   , **argv, required=True, type=str, help="*.txt file with the output")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    # Load reference structure (only one structure)
    print("\tReading data from '{:s}' ... ".format(args.input), end="")
    data_df = pd.read_csv(args.input, sep='\s+').set_index(['structure', 'unit_cell'])
    # data_df = pd.read_csv(
    #     args.input,
    #     sep='\s+',
    #     comment='#',
    #     header=None,
    #     names=['structure', 'unit_cell', 'dipole_x', 'dipole_y', 'dipole_z']
    # ).set_index(['structure', 'unit_cell'])
    print("done")

    #------------------#
    # Load full dataset (many structures)
    print("\tReading reference data from '{:s}' ... ".format(args.reference), end="")
    ref_df = pd.read_csv(args.reference, sep='\s+').set_index(['structure', 'unit_cell'])
    # ref_df = pd.read_csv(
    #     args.reference,
    #     sep='\s+',
    #     comment='#',
    #     header=None,
    #     names=['structure', 'unit_cell', 'dipole_x', 'dipole_y', 'dipole_z']
    # ).set_index(['structure', 'unit_cell'])
    print("done")
    
    #------------------#
    # Subtract reference structure (broadcasted by unit_cell)
    print("\tComputing the difference ... ", end="")
    reference = ref_df.loc[0]
    result = data_df.copy()
    for coord in ['dipole_x', 'dipole_y', 'dipole_z']:
        result[coord] = data_df[coord] - reference[coord]
    result = result.reset_index()
    print("done")

    #------------------# 
    print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
    # header = \
    #         f"Col 1: structure index\n" +\
    #         f"Col 2: unit-cell index\n" +\
    #         f"Col 3: dipole_x\n" +\
    #         f"Col 4: dipole_y\n"+\
    #         f"Col 5: dipole_z"
    # np.savetxt(args.output, result.to_numpy(), fmt=["%8d","%8d",float_format,float_format,float_format], header=header)
    header = ''.join(f"{col:>24s}" for col in ['structure', 'unit_cell', 'dipole_x', 'dipole_y', 'dipole_z'])
    np.savetxt(args.output,
        result.to_numpy(),
        fmt=["%24d", "%24d"] + [float_format]*3,
        header=header,comments="")
    print("done")
   
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
    