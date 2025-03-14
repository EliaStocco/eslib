import numpy as np
from typing import List, Tuple, Set
from ase import Atoms
import os

def check_atoms_consistency(atoms_list: List[Atoms])->Tuple[Set,Set]:
    """
    Verify that all ASE Atoms objects in the list have the same 'arrays' and 'info' keys.

    Parameters:
      atoms_list : List[Atoms]
          List of ASE Atoms objects.

    Returns:
      Tuple[set, set]
          A tuple containing:
            - The set of keys for the arrays (from the first Atoms object).
            - The set of keys for the info dictionary.

    Raises:
      ValueError: If any Atoms object has a different set of keys.
    """
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
    Save a list of ASE Atoms objects in a memory-efficient way to a compressed NPZ file.

    The file must have a '.npz' extension. The saved data includes:

      Global Metadata:
        - num_snapshots: Total number of snapshots.
        - num_atoms: 1D array with the number of atoms per snapshot.
        - pbc: 2D array (num_snapshots x 3) with periodic boundary conditions.
        - cell: 3D array (num_snapshots x 3 x 3) with cell matrices.

      Info Data:
        - For each info key in the Atoms objects, an array is saved with the key '_info_<key>'.
          These are assumed to be scalar values per snapshot (saved as 1D arrays).

      Array Data:
        - For each array key in the Atoms objects, the data is collected from all snapshots.
          Each per-snapshot array is stacked vertically into a single 2D array, saved with key '_array_<key>'.
          If an array is 1D, it is converted to 2D (by adding a new axis) prior to stacking.
          The stacked array's first dimension is the sum of atoms across all snapshots.

    Parameters:
      filename : str
          The filename for saving the data. Must have a ".npz" extension.
      atoms_list : List[Atoms]
          List of ASE Atoms objects to save.
      parallel : bool, optional
          (Currently unused) Flag to enable parallel processing.

    Raises:
      ValueError: If the file extension is not ".npz" or if any inconsistency is found.
    """
    # Determine the proper extension.
    root, ext = os.path.splitext(filename)
    if ext.lower() not in [".npz"]:
        raise ValueError(f"'{ext}' file extension not supported")
    
    ref_arrays, ref_info = check_atoms_consistency(atoms_list)
    
    # Global metadata: number of atoms, snapshots, pbc, and cell.
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
        assert value.shape[0] == num_snapshots, "error: info shape mismatch"
        to_save[key] = value
        
    # Process each arrays key.
    for name in ref_arrays:
        key = f"_array_{name}"
        # Collect the per-snapshot arrays.
        value:List[np.ndarray] = [atoms.arrays[name] for atoms in atoms_list]
        dims = np.asarray([v.ndim for v in value])
        # Ensure all arrays have the same number of dimensions.
        assert np.all(dims == dims[0]), "error: array dimension mismatch"
        # If the array is 1D, convert to 2D to allow stacking.
        if dims[0] == 1:
            value = [v[:, None] for v in value]
        # Stack all snapshots vertically.
        value:np.ndarray = np.vstack(value)
        # The total rows of the stacked array must equal the sum of atoms.
        assert value.shape[0] == np.sum(num_atoms), "error: stacked array shape mismatch"
        to_save[key] = value
    
    np.savez_compressed(root, **to_save)
    
def read_npz(filename: str) -> List[Atoms]:
    """
    Read a compressed NPZ file created with write_npz and reconstruct the list of ASE Atoms objects.

    The NPZ file must contain the following entries:

      Global Metadata:
        - num_snapshots: Total number of snapshots.
        - num_atoms: 1D array of the number of atoms per snapshot.
        - pbc: 2D array (num_snapshots x 3) of periodic boundary conditions.
        - cell: 3D array (num_snapshots x 3 x 3) of cell matrices.

      Info Data:
        - Each info key is stored with the prefix '_info_' as a 1D array (one value per snapshot).

      Array Data:
        - Each array key is stored with the prefix '_array_' as a 2D array, which is a vertical 
          stacking of all snapshots' data. The first dimension of this array is equal to the sum of atoms
          across all snapshots.

    This function splits each stacked array using the cumulative sum of num_atoms and assigns the
    corresponding info and arrays to new ASE Atoms objects. Special handling is applied for the
    "positions" key, which is used to initialize the Atoms object.

    Parameters:
      filename : str
          The filename of the compressed NPZ file.

    Returns:
      List[Atoms]
          A list of reconstructed ASE Atoms objects.

    Raises:
      ValueError: If the file cannot be read or if required keys are missing.
    """
    data = np.load(filename, allow_pickle=True)

    # Extract global metadata.
    num_snapshots = int(data["num_snapshots"])
    num_atoms = data["num_atoms"]  # 1D array, shape (num_snapshots,)
    pbc = data["pbc"]              # shape (num_snapshots, 3)
    cell = data["cell"]            # shape (num_snapshots, 3, 3)

    # Identify the keys for info and arrays.
    info_keys:List[str] = [key for key in data.keys() if key.startswith("_info_")]
    array_keys:List[str] = [key for key in data.keys() if key.startswith("_array_")]
    
    if "_array_positions" not in array_keys:
        raise ValueError("'positions' not found in file.")

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
        # Split the stacked array into pieces; use cumsum[:-1] as split indices.
        per_snapshot_arrays = np.split(stacked_array, cumsum[:-1], axis=0)
        arrays_dict[orig_key] = per_snapshot_arrays

    # Reconstruct each Atoms object.
    atoms_list = [None]*num_snapshots
    for i in range(num_snapshots):
        # Use the "positions" array (if available) to initialize the Atoms object.
        pos = np.asarray(arrays_dict["positions"][i])
        atoms = Atoms(cell=cell[i], pbc=pbc[i], positions=pos)
        # Assign the remaining arrays (other than "positions").
        for key, arrays in arrays_dict.items():
            if key == "positions":
                continue
            arr = np.asarray(arrays[i])
            # If the stored array was originally 1D (converted to 2D with one column), flatten it.
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()
            atoms.set_array(key, arr)
        # Assign the info values.
        for key, values in info_dict.items():
            atoms.info[key] = values[i]
        atoms_list[i] = atoms
        
    return atoms_list
