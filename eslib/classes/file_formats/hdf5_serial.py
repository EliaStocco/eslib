"""
HDF5 input/output functions for ASE Atoms objects.

Functions:
read_hdf5: Read a list of Atoms objects from an HDF5 file.
write_hdf5: Write a list of Atoms objects to an HDF5 file.

Examples:
>>> from eslib.classes.hdf5 import read_hdf5, write_hdf5
>>> from ase import Atoms

>>> # Create a list of Atoms objects
>>> atoms_list = [Atoms('H2O', positions=[[0, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0.1]]) for _ in range(5)]

>>> # Write the list of Atoms objects to an HDF5 file
>>> write_hdf5('atoms.hdf5', atoms_list)

>>> # Read the list of Atoms objects from the HDF5 file
>>> read_atoms_list = read_hdf5('atoms.hdf5')
>>> print(len(read_atoms_list))
5
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List

import h5py
import numpy as np
from ase import Atoms


def read_hdf5(filename: str, index: slice = None) -> List[Atoms]:
    """
    Read a list of Atoms objects from an HDF5 file, allowing for slicing.

    Parameters:
    filename (str): The name of the HDF5 file to read from.
    index (slice): A slice object to select a subset of the structures.

    Returns:
    List[Atoms]: A list of Atoms objects read from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        atoms_group = f['atoms']
        num_structures = len(atoms_group)
        
        # Apply slicing if provided
        atom_keys = list(atoms_group.keys())  # get all keys (which are typically strings)
        if index is not None:
            atom_keys = atom_keys[index]  # Apply the slice to the keys
        
        atoms_list = [None] * len(atom_keys)
        
        # Iterate over the selected Atoms objects
        for n, atom_key in enumerate(atom_keys):
            atom_group = atoms_group[atom_key]

            # Extract the stored positions, atomic numbers, cell, and PBC
            positions = np.asarray(atom_group['positions'][:])
            numbers = np.asarray(atom_group['atomic_numbers'][:])
            cell = np.asarray(atom_group['cell'][:])
            pbc = np.asarray(atom_group['pbc'][:])

            # Create the Atoms object
            atoms = Atoms(positions=positions, numbers=numbers, cell=cell, pbc=pbc)
            
            # Extract additional info and arrays
            for key in atom_group.keys():
                key = str(key)
                if key in ["positions", "atomic_numbers", "cell", "pbc"]:
                    continue
                elif "_info_" in key:
                    _key = key.replace("_info_", "")
                    atoms.info[_key] = np.asarray(atom_group[key])
                    if atoms.info[_key].ndim == 0:
                        atoms.info[_key] = float(atoms.info[_key])
                elif "_arrays_" in key:
                    _key = key.replace("_arrays_", "")
                    atoms.arrays[_key] = np.asarray(atom_group[key])
                else:
                    raise ValueError(f"Unknown key: {key}")
            
            atoms_list[n] = atoms
        
        return atoms_list


def write_hdf5(filename: str, atoms_list: List[Atoms]) -> None:
    """
    Write a list of Atoms objects to an HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file to write to.
    atoms_list (List[Atoms]): A list of Atoms objects to write to the HDF5 file.
    """

    # Create an HDF5 file
    with h5py.File(filename, 'w') as f:
        # Create a group to store the Atoms objects
        # f.create_dataset("num_atoms", data=[a.get_global_number_of_atoms() for a in atoms_list])
        atoms_group = f.create_group('atoms')

        # Iterate over the list of Atoms objects
        for i, atoms in enumerate(atoms_list):
            # Create a subgroup for each Atoms object
            atom_group = atoms_group.create_group(f'atom_{i}')

            # Store the positions, atomic numbers, and cell of the Atoms object
            positions = atoms.get_positions()
            numbers = atoms.get_atomic_numbers()  # Use atomic numbers for more consistency
            cell = atoms.get_cell().array
            pbc = atoms.get_pbc()

            # Create datasets to store the positions, atomic numbers, cell, and PBC
            atom_group.create_dataset('positions', data=positions)
            atom_group.create_dataset('atomic_numbers', data=numbers)
            atom_group.create_dataset('cell', data=cell)
            atom_group.create_dataset('pbc', data=pbc)
            
            info = atoms.info.keys()
            for i in info:
                atom_group.create_dataset("_info_{:s}".format(i), data=atoms.info[i])
                
            arrays = atoms.arrays.keys()
            for a in arrays:
                if a in ["positions","numbers"] :
                    continue
                atom_group.create_dataset("_arrays_{:s}".format(a), data=atoms.arrays[a])

