from copy import copy
from typing import List, Tuple, Union

import numpy as np

from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.tools import cart2frac, frac2cart, is_integer


def modular_norm(numbers: np.ndarray, modulus: float = 1, threshold: float = 0.01) -> np.ndarray:
    """
    Calculate the modular norm between a list of numbers and a given modulus.

    Parameters:
    numbers (List[float]): A list of numbers.
    modulus (float, optional): The modulus value. Defaults to 1.
    threshold (float, optional): The threshold value for the distances. Defaults to 0.01.

    Returns:
    numpy.ndarray: An array of distances between the numbers and the modulus.
    """
    numbers = np.mod(numbers, modulus)
    distances = np.minimum(numbers % 1, 1 - (numbers % 1))  # Map to closest distance to 0
    return distances

def angle_between_vectors(array1: List[float], array2: List[float]) -> float:
    """
    Compute the angle in radians between two vectors of the same length.

    Parameters:
    array1 (List[float]): First vector.
    array2 (List[float]): Second vector.

    Returns:
    float: Angle in radians between the two vectors.
    """
    dot_product = np.dot(array1, array2)
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    cosine_similarity = dot_product / (norm1 * norm2)
    angle_radians = np.arccos(cosine_similarity)
    return angle_radians



def fold(trajectory:AtomicStructures)->Tuple[AtomicStructures,np.ndarray]:
    """Fold a trajectory onto the primitive unit cell."""
    
    folded = copy(trajectory)
    shift = [None]*len(trajectory)
    for n,atoms in enumerate(trajectory):
        frac_positions = cart2frac(atoms.get_cell(),atoms.positions)
        folded_positions = np.mod(frac_positions,1)
        if np.any(np.abs(folded_positions)>1):
            raise ValueError("coding error")
        shift[n] = folded_positions - frac_positions
        if np.any( [ not is_integer(i) for i in shift[n].flatten()] ):
            raise ValueError("coding error")
        positions = frac2cart(atoms.get_cell(),folded_positions)
        folded[n].positions = positions

    return folded, np.asarray(shift)

def mic_dist(dr: np.ndarray, cells: Union[np.ndarray, Cell, List[Cell]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Minimum Image Convention (MIC) to displacement vectors.

    Parameters
    ----------
    dr : (natoms,3) or (nframes,natoms,3) ndarray
        Displacement vectors.
    cells : ase.Cell, list[ase.Cell], or ndarray
        Cell(s) for each frame:
        - single Cell or (3,3) ndarray → used for all frames
        - list of Cells → one per frame

    Returns
    -------
    dr_mic : ndarray
        MIC-corrected displacement vectors (same shape as dr).
    dist : ndarray
        Distances corresponding to dr_mic:
        - shape (natoms,) if dr was (natoms,3)
        - shape (nframes,natoms) if dr was (nframes,natoms,3)
    """
    dr = np.asarray(dr)

    # --- promote dr to (nframes,natoms,3) ---
    if dr.ndim == 2:  # (natoms,3)
        dr = dr[None, :, :]
        single_frame = True
    elif dr.ndim == 3:
        single_frame = False
    else:
        raise ValueError("dr must have shape (natoms,3) or (nframes,natoms,3)")

    nframes, natoms, _ = dr.shape

    # --- normalize cells ---
    if isinstance(cells, Cell):
        cells = np.broadcast_to(cells.array, (nframes, 3, 3))
    elif isinstance(cells, list) and all(isinstance(c, Cell) for c in cells):
        cells = np.array([c.array.T for c in cells])
    else:
        cells = np.asarray(cells)
        if cells.ndim == 2:  # single cell matrix
            cells = np.broadcast_to(cells, (nframes, 3, 3))

    # --- inverse cells (for fractional coords) ---
    inv_cells = np.linalg.inv(cells)  # (nframes,3,3)

    # --- convert dr -> fractional coordinates ---
    dr_frac = np.einsum("fij,fkj->fki", inv_cells, dr)

    # --- apply MIC: wrap into [-0.5,0.5) ---
    dr_frac -= np.round(dr_frac)

    # --- back to Cartesian ---
    dr_mic = np.einsum("fij,fkj->fki", cells, dr_frac)

    # --- distances ---
    dist = np.linalg.norm(dr_mic, axis=-1)

    if single_frame:
        return dr_mic[0], dist[0]
    else:
        return dr_mic, dist

def max_mic_distance(cell: Cell) -> float:
    """
    Compute the maximum distance that can be computed with the minimum image convention (MIC).
    
    Parameters
    ----------
    cell : ase.cell.Cell
        The unit cell object containing lattice vectors.
    
    Returns
    -------
    float
        Maximum possible distance between two atoms considering PBC.
    """
    # Get the lattice vectors
    a, b, c = cell.array  # shape (3,3)
    
    # Compute the "body diagonal" vector (sum of 0.5 * each vector)
    diag_vec = 0.5 * (a + b + c)
    
    # Maximum MIC distance is the length of this vector
    return np.linalg.norm(diag_vec)