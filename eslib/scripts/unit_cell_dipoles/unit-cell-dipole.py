#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog, float_format
from eslib.mathematics import melt

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dipole of the unitcells in a supercell."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="input file")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-ad", "--atomic_dipoles", **argv, required=True , type=str, help="atomic dipoles keyword")
    parser.add_argument("-d" , "--dipole"        , **argv, required=False, type=str, help="dipole keyword (default: %(default)s)",default=None)
    parser.add_argument("-uc", "--unit_cell"     , **argv, required=True , type=str, help="unit-cell keyword")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="*.txt output file with unit-cells dipoles")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of strctures: ",len(structures))
    print("\tn. of atoms: ",structures[0].get_global_number_of_atoms())
    
    #------------------#
    with eslog("\nExtracting data"):
        
        atomic_dipoles = structures.get_array(args.atomic_dipoles)
        if args.dipole is not None:
            dipole = structures.get_info(args.dipole)
            sum_dipole = np.sum(atomic_dipoles,axis=1)
            assert np.allclose(sum_dipole,dipole), \
                f"Sum of atomic dipoles ({sum_dipole}) does not match the dipole ({dipole})"
        
        unit_cell = structures.get_array(args.unit_cell)
    
    #------------------#
    with eslog("\nCreating atomic dipoles dataframe"):
        testAD = melt(atomic_dipoles,name="atomic_dipoles")
        
    with eslog("Creating unit cells dataframe"):
        testUC = melt(unit_cell.astype(int),name="unit_cell")
        
    with eslog("Merging dataframes"):
        data = pd.merge(testAD, testUC)

    #------------------#
    with eslog("Preparing results for output"):
        data = data.rename(columns={
            'dim_0': 'structure',
            'dim_1': 'atom',
            'dim_2': 'component',
        })

        data = (
            data
            .groupby(['structure', 'component', 'unit_cell'], as_index=False)
            .agg(dipole=('atomic_dipoles', 'sum'))
        )
        
        data = data.pivot_table(index=['structure', 'unit_cell'], columns='component', values='dipole', aggfunc='first')
        data.columns = ['dipole_x', 'dipole_y', 'dipole_z']
        data.reset_index(inplace=True)
        
    #------------------# 
    print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
    header = \
            f"Col 1: structure index\n" +\
            f"Col 2: unit-cell index\n" +\
            f"Col 3: dipole_x\n" +\
            f"Col 4: dipole_y\n"+\
            f"Col 5: dipole_z"
    np.savetxt(args.output, data.to_numpy(), fmt=["%8d","%8d",float_format,float_format,float_format], header=header)
    print("done")
    
    return
    
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