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

import h5py
import numpy as np
from ase import Atoms
from typing import List
from concurrent.futures import ThreadPoolExecutor

def read_atoms_from_group(atom_group) -> Atoms:
    """ Helper function to read a single Atoms object from a group """
    # Extract the stored positions, atomic numbers, cell, and PBC
    positions = atom_group['positions'][:]
    numbers = atom_group['atomic_numbers'][:]
    cell = atom_group['cell'][:]
    pbc = atom_group['pbc'][:]

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
    return atoms

def read_hdf5(filename: str, index: slice = None, parallel: bool = True) -> List[Atoms]:
    """
    Read a list of Atoms objects from an HDF5 file, allowing for slicing and parallel processing.

    Parameters:
    filename (str): The name of the HDF5 file to read from.
    index (slice): A slice object to select a subset of the structures.
    parallel (bool): If True, read the atoms in parallel using threads.

    Returns:
    List[Atoms]: A list of Atoms objects read from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        atoms_group = f['atoms']
        atom_keys = list(atoms_group.keys())  # get all keys (which are typically strings)
        
        # Apply slicing if provided
        if index is not None:
            atom_keys = atom_keys[index]  # Apply the slice to the keys
        
        # Sequential reading
        if not parallel:
            atoms_list = [read_atoms_from_group(atoms_group[atom_key]) for atom_key in atom_keys]
        # Parallel reading (thread-based)
        else:
            with ThreadPoolExecutor() as executor:
                atoms_list = list(executor.map(lambda key: read_atoms_from_group(atoms_group[key]), atom_keys))
        
        return atoms_list

def write_single_atom_group(atoms, i):
    """ Helper function to prepare data for a single Atoms object """
    # Prepare all the necessary data for each Atoms object
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()  # Use atomic numbers for more consistency
    cell = atoms.get_cell().array
    pbc = atoms.get_pbc()

    # Collect info and arrays
    info = {f"_info_{key}": atoms.info[key] for key in atoms.info}
    arrays = {f"_arrays_{key}": atoms.arrays[key] for key in atoms.arrays if key not in ["positions", "numbers"]}
    
    return (i, positions, numbers, cell, pbc, info, arrays)

def write_hdf5(filename: str, atoms_list: List[Atoms], parallel: bool = True) -> None:
    """
    Write a list of Atoms objects to an HDF5 file with parallel processing.

    Parameters:
    filename (str): The name of the HDF5 file to write to.
    atoms_list (List[Atoms]): A list of Atoms objects to write to the HDF5 file.
    parallel (bool): If True, parallelize the data preparation steps.
    """

    # Prepare the data in parallel if required
    if parallel:
        with ThreadPoolExecutor() as executor:
            prepared_data = list(executor.map(write_single_atom_group, atoms_list, range(len(atoms_list))))
    else:
        prepared_data = [write_single_atom_group(atoms, i) for i, atoms in enumerate(atoms_list)]

    # Create the HDF5 file and write the data (this part cannot be parallelized)
    with h5py.File(filename, 'w') as f:
        # Create a group to store the Atoms objects
        atoms_group = f.create_group('atoms')

        # Write the data to the HDF5 file
        for data in prepared_data:
            i, positions, numbers, cell, pbc, info, arrays = data
            atom_group = atoms_group.create_group(f'atom_{i}')

            # Store the positions, atomic numbers, cell, and PBC
            atom_group.create_dataset('positions', data=positions)
            atom_group.create_dataset('atomic_numbers', data=numbers)
            atom_group.create_dataset('cell', data=cell)
            atom_group.create_dataset('pbc', data=pbc)

            # Store the info and arrays
            for key, value in info.items():
                atom_group.create_dataset(key, data=value)
            
            for key, value in arrays.items():
                atom_group.create_dataset(key, data=value)
