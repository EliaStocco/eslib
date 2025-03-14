import numpy as np
from typing import List
from ase import Atoms
import os
import sys

def check_atoms_consistency(atoms_list: List[Atoms]):
    """Check if all Atoms objects have the same arrays and info keys."""
    if not atoms_list:
        print("The list is empty.")
        return False  # Consider raising an exception instead.
    
    # Use the first Atoms object as a reference.
    ref_atoms = atoms_list[0]
    ref_arrays = set(ref_atoms.arrays.keys())
    ref_info = set(ref_atoms.info.keys())

    # Verify that each subsequent Atoms object has the same keys.
    for i, atoms in enumerate(atoms_list[1:], start=1):
        if set(atoms.arrays.keys()) != ref_arrays:
            raise ValueError(f"Mismatch in arrays at index {i}: {set(atoms.arrays.keys())} vs {ref_arrays}")
        if set(atoms.info.keys()) != ref_info:
            raise ValueError(f"Mismatch in info keys at index {i}: {set(atoms.info.keys())} vs {ref_info}")
    return ref_arrays, ref_info

def write_npz(filename: str, atoms_list: List[Atoms], parallel: bool = True) -> None:
    """
    Save a list of ASE Atoms objects in a memory-efficient way.
    
    The metadata saved includes:
      - Number of snapshots and atoms per snapshot.
      - PBC and cell for each snapshot.
      - Info values stored as 1D arrays.
      - Arrays stored as 2D arrays (each snapshot's data is stacked vertically).
      
    """
    # Determine the proper extension.
    root, ext = os.path.splitext(filename)
    if ext.lower() not in [".npz"]:
        raise ValueError(f"'{ext}' file extension not supported")
    
    ref_arrays, ref_info = check_atoms_consistency(atoms_list)
    
    # Global metadata: number of atoms, snapshots, PBC, and cell.
    num_atoms = np.asarray([len(atoms) for atoms in atoms_list]).astype(int)
    num_snapshots = len(num_atoms)
    pbc = np.asarray([atoms.get_pbc() for atoms in atoms_list]).astype(bool)
    cell = np.asarray([np.asarray(atoms.get_cell()) for atoms in atoms_list])
    assert pbc.shape == (num_snapshots, 3), "error: pbc shape mismatch"
    assert cell.shape == (num_snapshots, 3, 3), "error: cell shape mismatch"
    
    # Container for all data to be saved.
    to_save = {
        "num_snapshots": num_snapshots,
        "num_atoms": num_atoms,
        "pbc": pbc,
        "cell": cell,
    }
    
    # Process each info key (assumed scalar per snapshot).
    for name in ref_info:
        key = f"_info_{name}"
        value = np.asarray([atoms.info[name] for atoms in atoms_list])
        assert value.shape == (num_snapshots,), "error: info shape mismatch"
        to_save[key] = value
        
    # Process each arrays key.
    for name in ref_arrays:
        key = f"_array_{name}"
        # Collect the per-snapshot arrays.
        value = [atoms.arrays[name] for atoms in atoms_list]
        dims = np.asarray([v.ndim for v in value])
        # Ensure all arrays have the same number of dimensions.
        assert np.all(dims == dims[0]), "error: array dimension mismatch"
        # If the array is 1D, convert to 2D to allow stacking.
        if dims[0] == 1:
            value = [v[:, None] for v in value]
        # Stack all snapshots vertically.
        value = np.vstack(value)
        # The total rows of the stacked array must equal the sum of atoms.
        assert value.shape[0] == np.sum(num_atoms), "error: stacked array shape mismatch"
        to_save[key] = value
    
    np.savez_compressed(root, **to_save)
    
def read_npz(filename: str) -> List[Atoms]:
    """
    Read a file saved with write_npy and reconstruct the list of ASE Atoms objects.
    
    The file should contain:
      - Global metadata: "num_snapshots", "num_atoms", "pbc", "cell"
      - For each info key: stored as "_info_<key>" (1D array of length num_snapshots)
      - For each array key: stored as "_array_<key>" (a 2D stacked array;
        the number of rows equals sum(num_atoms) across snapshots)
    
    This function splits each stacked array using the cumulative sum of num_atoms,
    and assigns the corresponding info and arrays to new Atoms objects.
    """
    # Determine the proper file to load based on the extension.
    data = np.load(filename,allow_pickle=True)

    # Extract global metadata.
    num_snapshots = int(data["num_snapshots"])
    num_atoms = data["num_atoms"]  # 1D array, shape (num_snapshots,)
    pbc = data["pbc"]              # shape (num_snapshots, 3)
    cell = data["cell"]            # shape (num_snapshots, 3, 3)

    # Identify the keys for info and arrays.
    info_keys = [key for key in data.keys() if key.startswith("_info_")]
    array_keys = [key for key in data.keys() if key.startswith("_array_")]

    # Reconstruct per-snapshot info values.
    info_dict = {}
    for key in info_keys:
        orig_key = key.replace("_info_", "", 1)
        info_arr = data[key]  # Expected shape: (num_snapshots,)
        # Convert to a list of scalars (one per snapshot).
        info_dict[orig_key] = [info_arr[i] for i in range(num_snapshots)]

    # Reconstruct per-snapshot arrays.
    arrays_dict = {}
    for key in array_keys:
        orig_key = key.replace("_array_", "", 1)
        stacked_array = data[key]  # 2D array with shape (sum(num_atoms), d)
        # Compute the cumulative sum of num_atoms to determine splitting indices.
        cumsum = np.cumsum(num_atoms)
        # Split the stacked array into pieces. np.split uses indices for where to split;
        # we use cumsum[:-1] so that each piece corresponds to one snapshot.
        per_snapshot_arrays = np.split(stacked_array, cumsum[:-1], axis=0)
        arrays_dict[orig_key] = per_snapshot_arrays

    # Now, reconstruct each Atoms object.
    atoms_list = []
    for i in range(num_snapshots):
        # Create an Atoms object using the stored cell and pbc.
        pos = np.asarray(arrays_dict["positions"][i])
        atoms = Atoms(cell=cell[i], pbc=pbc[i],positions=pos)
        # Assign the info values.
        
        # Assign the arrays.
        for key, arrays in arrays_dict.items():
            if key == "positions":
                continue
            arr = arrays[i]
            # If the stored array was originally 1D (converted to 2D with one column),
            # flatten it back.
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()
            
            atoms.set_array(key, arr)
            
        for key, values in info_dict.items():
            atoms.info[key] = values[i]
            
            
        atoms_list.append(atoms)
        
    return atoms_list