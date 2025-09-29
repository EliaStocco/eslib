#!/usr/bin/env python
import numpy as np
import pandas as pd
from itertools import chain
from multiprocessing import Process, Manager, cpu_count
from ase.cell import Cell
from ase import Atoms
from typing import List
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog, warning, float_format
from eslib.tools import cart2frac
from eslib.input import flist
from eslib.mathematics import dcast, melt

#---------------------------------------#
description = "Analyse the unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-d" , "--dipoles"     , **argv, required=True , type=str  , help="*.txt file with the results produced by 'unit-cell-dipole.py'")
    parser.add_argument("-v" , "--vector"      , **argv, required=False, type=flist, help="vector along which evalute the projection [factional] (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"      , **argv, required=True , type=str  , help="*.txt output file with the analysed unit-cells dipoles")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #------------------#
    with eslog(f"Reading the first atomic structure from file '{args.input}'"):
        structures:List[Atoms] = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("\tn. of structures: ", len(structures))
    print("\tn. of atoms:     ", structures[0].get_global_number_of_atoms())

    #------------------#
    print("\tReading dipoles from '{:s}' ... ".format(args.dipoles), end="")
    # dipoles = np.loadtxt(args.dipoles)
    # dipoles = pd.DataFrame(dipoles,
    #                        columns=["structure", "unit_cell",
    #                                 "dipole_x", "dipole_y", "dipole_z"])
    dipoles = pd.read_csv(args.dipoles, sep='\s+')
    dipoles["structure"] = dipoles["structure"].astype(int)
    dipoles["unit_cell"]  = dipoles["unit_cell"].astype(int)
    
    dipole_array = dcast(dipoles, ["structure","unit_cell"])

    print("done")

    #------------------#
    # atoms = structures[0]
    # super_cell = atoms.get_cell()
    sub_df = dipoles[dipoles["structure"] == 0]
    n_unit_cells = len(sub_df)
    size = int(round(n_unit_cells ** (1./3)))
    print(f"\n\t{warning}:\n\t"
          f"We deduced a supercell of size {size}x{size}x{size}.\n")
    # unit_cell = Cell(np.asarray(super_cell) / size)
    
    #------------------#
    with eslog("Extracting unit-cells"):
        unit_cells = np.asarray([ atoms.get_cell().array.T for atoms in structures ])/size
        
    #------------------#
    print("\n\t  dipoles.shape: ", dipole_array.shape)
    print("\t unit_cells.shape: ", unit_cells.shape)
    
    #------------------#
    with eslog("Computing the fractional coordinates of the dipoles"):
        original_shape = unit_cells.shape[:-2]
        reshaped = unit_cells.reshape(-1, 3, 3)
        inverted = np.linalg.inv(reshaped)
        inverted = inverted.reshape(*original_shape, 3, 3)
        frac = np.asarray(np.einsum('aij,abj->abi', inverted, dipole_array))
        assert frac.shape == dipole_array.shape, "Result shape mismatch!"
    
    with eslog("Merging dataframes"):
        frac = melt(frac, index={0: "structure", 1: "unit_cell"}, value_names=["frac_1", "frac_2", "frac_3"])
        assert frac.shape == dipoles.shape, "Result shape mismatch!"
        data = pd.merge(dipoles, frac, on=["structure","unit_cell"], how='inner')
        
    #------------------#
    with eslog("Computing extra information"):
        data["dipole_norm"] = np.linalg.norm(data[["dipole_x", "dipole_y", "dipole_z"]].to_numpy(), axis=1)
        
    #------------------#s
    column_names = [
        "structure", "unit_cell", "dipole_x", "dipole_y", "dipole_z",
        "frac_1", "frac_2", "frac_3", "dipole_norm"
    ]
    header = ''.join(f"{col:>24s}" for col in column_names)
    
    if args.vector is None:
        N = 7
    else:
        N = 8
        header += f"{'dipole_projection':>24s}"

        vector = np.asarray(args.vector)
        vectors = np.asarray(np.einsum('aij,j->ai',unit_cells, vector))
        assert vectors.shape == (len(structures), 3), "Result shape mismatch!"
        vectors = vectors/np.linalg.norm(vectors, axis=1)[:,None]
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0), "Vectors are not unit vectors!"
        vectors = vectors[:,None,:] # add unit-cell
        projection = np.asarray(np.einsum('abi,abi->ab',dipole_array, vectors))
        projection = melt(projection,index={0: "structure", 1: "unit_cell"}, value_names=["dipole_projection"])
        data = pd.merge(data, projection, on=["structure","unit_cell"], how='inner')
        
    #------------------#
    with eslog(f"Saving results to file '{args.output}' ... "):
        np.savetxt(args.output,
                data.to_numpy(),
                fmt=["%24d", "%24d"] + [float_format]*N,
                header=header,comments="")

#---------------------------------------#
if __name__ == "__main__":
    main()

#------------------#
# dipoles["dipole_1"] = np.nan
# dipoles["dipole_2"] = np.nan
# dipoles["dipole_3"] = np.nan

# if MULTIPROCESSING:
#     # 1) split structures for worker processes
#     num_chunks = min(cpu_count(), 8)
#     struct_chunks = np.array_split(range(len(structures)), num_chunks)

#     manager = Manager()
#     return_dict = manager.dict()
#     procs = []
#     for i, chunk in enumerate(struct_chunks):
#         # slice the dipoles DataFrame for this chunk
#         dipoles_chunk = dipoles[dipoles["structure"].isin(chunk)]
#         p = Process(target=worker_chunk,
#                     args=(i, chunk, dipoles_chunk,
#                           unit_cell, return_dict))
#         p.start()
#         procs.append(p)
#     for p in procs:
#         p.join()

#     # 2) gather all results
#     all_results = list(chain.from_iterable(return_dict.values()))
#     if not all_results:
#         raise RuntimeError("No results produced by worker_chunk!")

#     # 3) sequential merge back into dipoles with tqdm
#     for idxs, frac in tqdm(all_results, desc="Merging results", mininterval=0.5):
#         dipoles.loc[idxs, "dipole_1"] = frac[:, 0]
#         dipoles.loc[idxs, "dipole_2"] = frac[:, 1]
#         dipoles.loc[idxs, "dipole_3"] = frac[:, 2]

# else:
#     # Single‐process fallback
#     for n, atoms in tqdm(enumerate(structures),
#                           total=len(structures),
#                           desc="Processing structures",
#                           mininterval=0.5):
#         sub_df = dipoles[dipoles["structure"] == n]
#         mu = sub_df[["dipole_x", "dipole_y", "dipole_z"]].to_numpy()
#         frac = cart2frac(cell=unit_cell, v=mu)
#         dipoles.loc[sub_df.index, "dipole_1"] = frac[:, 0]
#         dipoles.loc[sub_df.index, "dipole_2"] = frac[:, 1]
#         dipoles.loc[sub_df.index, "dipole_3"] = frac[:, 2]

# #------------------#
# dipoles["dipole_norm"] = np.linalg.norm(dipoles[["dipole_x", "dipole_y", "dipole_z"]].to_numpy(), axis=1)
# print("\n\tDipole norm summary:")
# print("\tdipole min: ", dipoles["dipole_norm"].min())
# print("\tdipole max: ", dipoles["dipole_norm"].max())

# if args.vector is not None:
#     # Projection along a vector
#     vector = np.loadtxt(args.vector)
#     vector = vector / np.linalg.norm(vector)
#     dipoles["projection"] = np.dot(dipoles[["dipole_x", "dipole_y", "dipole_z"]].to_numpy(), vector)
    
# else:
#     # Projection along the dipole
#     dipoles["projection"] = np.nan

#---------------------------------------#
# def worker_chunk(chunk_id, structure_ids, dipoles, unit_cell, return_dict):
#     local_result = [None]*len(structure_ids)
#     k = 0
#     for n in tqdm(structure_ids,
#                   position=chunk_id,
#                   desc=f"Worker {chunk_id}",
#                   leave=True,
#                   mininterval=0.5):
#         sub_df = dipoles[dipoles["structure"] == n]
#         if sub_df.empty:
#             continue
#         mu = sub_df[["dipole_x", "dipole_y", "dipole_z"]].to_numpy()
#         frac = np.atleast_2d(cart2frac(cell=unit_cell, v=mu))
#         local_result[k] = (sub_df.index.to_numpy(), frac)
#         k += 1
#     assert len(local_result) == len(structure_ids), "Not all results were produced!"
#     return_dict[chunk_id] = local_result