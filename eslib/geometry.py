from classes.atomic_structures import AtomicStructures
from eslib.tools import cart2frac, frac2cart
import numpy as np
from copy import copy
from eslib.tools import is_integer
from typing import Tuple
from typing import List

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
