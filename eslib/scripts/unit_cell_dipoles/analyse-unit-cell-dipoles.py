#!/usr/bin/env python
import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog, warning, float_format
from eslib.tools import cart2frac


#---------------------------------------#
description = "Analyse the unit-cells dipoles."
MULTIPROCESSING = False

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-d" , "--dipoles"       , **argv, required=True , type=str, help="*.txt file with the results produced by 'unit-cell-dipole.py'")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="*.txt output file with the analysed unit-cells dipoles")
    return parser

#---------------------------------------#
if MULTIPROCESSING:
    
    from multiprocessing import Pool, cpu_count, Value, Lock
    
    num_cores = 4 # cpu_count()
    
    # Shared variables for progress tracking
    progress = None
    lock = None
    
    # Helper function to split a list into chunks
    def split_into_chunks(data, num_chunks):
        """Split a list into num_chunks approximately equal parts."""
        chunk_size = len(data) // num_chunks
        remainder = len(data) % num_chunks
        chunks = []
        start = 0
        for i in range(num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(data[start:end])
            start = end
        return chunks

    # Function to process a chunk of structures
    def process_chunk(chunk_args):
        global progress, lock
        chunk, dipoles, unit_cell = chunk_args
        results = []
        for n, atoms in chunk:
            sub_df = dipoles[dipoles["structure"] == n]
            mu = sub_df[["dipole_x", "dipole_y", "dipole_z"]].to_numpy()
            frac = cart2frac(cell=unit_cell, v=mu)
            
            result = {
                "index": sub_df.index,
                "dipole_1": frac[:, 0],
                "dipole_2": frac[:, 1],
                "dipole_3": frac[:, 2],
            }
            results.append(result)
            
            # Update progress
            with lock:
                progress.value += 1
        return results

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading the first atomic structure from file '{args.input}'"):
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("\tn. of strctures: ",len(structures))
    print("\tn. of atoms: ",structures[0].get_global_number_of_atoms())
    
    #------------------#
    print("\tReading dipoles from '{:s}' ... ".format(args.dipoles), end="")
    dipoles = np.loadtxt(args.dipoles)
    dipoles = pd.DataFrame(dipoles,columns=["structure","unit_cell","dipole_x","dipole_y","dipole_z"])
    dipoles["structure"] = dipoles["structure"].astype(int)
    dipoles["unit_cell"] = dipoles["unit_cell"].astype(int)
    print("done")
    
    #------------------#
    atoms = structures[0]
    super_cell = atoms.get_cell()
    sub_df = dipoles[dipoles["structure"] == 0]
    n_unit_cells = len(sub_df)
    size = np.round(n_unit_cells ** (1./3)).astype(int)
    print(f"\n\t{warning}:\n\tWe deduced that you have provided a supercell of size {size}x{size}x{size}.\n\tBe sure that this is correct.\n")
    unit_cell = np.asarray(super_cell)/size
    unit_cell = Cell(unit_cell)
    
    #------------------#
    dipoles["dipole_1"] = None
    dipoles["dipole_2"] = None
    dipoles["dipole_3"] = None
    
    N = len(structures)
    # num_cores = cpu_count()
    
    if MULTIPROCESSING:
        
        global progress, lock
        
        # Divide structures into chunks
        chunks = split_into_chunks(list(enumerate(structures)), num_cores)
        pool_args = [(chunk, dipoles, unit_cell) for chunk in chunks]

        # Initialize shared variables for progress tracking
        progress = Value('i', 0)  # Shared integer for progress
        lock = Lock()
        
        # Preallocate chunk_results with the correct size
        chunk_results = [None] * len(pool_args)

        # Use multiprocessing to process chunks in parallel
        print(f"\tUsing {num_cores} CPU cores for parallel processing.")
        with Pool(processes=num_cores) as pool:
            # Use tqdm to display a progress bar
            with tqdm(total=len(pool_args), desc="Processing structures") as pbar:
                for i, result in enumerate(pool.imap_unordered(process_chunk, pool_args)):
                    chunk_results[i] = result  # Assign result to the preallocated list
                    with lock:
                        pbar.update(1)

        # Update the dipoles DataFrame with results
        with eslog(f"Assembling results"):
            for results in chunk_results:
                for result in results:
                    dipoles.loc[result["index"], "dipole_1"] = result["dipole_1"]
                    dipoles.loc[result["index"], "dipole_2"] = result["dipole_2"]
                    dipoles.loc[result["index"], "dipole_3"] = result["dipole_3"]
                
    else:
        
        for n, atoms in tqdm(enumerate(structures), total=len(structures), desc="Processing structures"):
            sub_df = dipoles[dipoles["structure"] == n]
            mu = sub_df[["dipole_x", "dipole_y", "dipole_z"]].to_numpy()
            frac = cart2frac(cell=unit_cell, v=mu)

            dipoles.loc[sub_df.index, "dipole_1"] = frac[:, 0]
            dipoles.loc[sub_df.index, "dipole_2"] = frac[:, 1]
            dipoles.loc[sub_df.index, "dipole_3"] = frac[:, 2]

    #------------------# 
    print("\n\tSaving results to file '{:s}' ... ".format(args.output), end="")
    header = \
            f"Col 1: structure index\n" +\
            f"Col 2: unit-cell index\n" +\
            f"Col 3: dipole_x\n" +\
            f"Col 4: dipole_y\n"+\
            f"Col 5: dipole_z\n"+\
            f"Col 6: frac_1\n" +\
            f"Col 7: frac_2\n"+\
            f"Col 8: frac_3"
    np.savetxt(args.output, dipoles.to_numpy(), fmt=["%8d","%8d"]+[float_format]*6, header=header)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
    