from eslib.classes.trajectory import AtomicStructures
from eslib.tools import cart2frac, frac2cart
import numpy as np
from copy import copy
from eslib.tools import is_integer
from typing import Tuple

def fold(trajectory:AtomicStructures)->Tuple[AtomicStructures,np.ndarray]:
    """Fold a trajectory onto the primitive unit cell."""
    
    folded = copy(trajectory)
    shift = [None]*len(trajectory)
    for n,atoms in enumerate(trajectory):
        frac_positions = cart2frac(atoms.get_cell(),atoms.positions)
        folded_positions = np.mod(frac_positions,1)
        if np.any(np.abs(folded_positions)>1):
            raise ValueError("coding error")
        shift[n] = frac_positions - folded_positions
        if np.any( [ not is_integer(i) for i in shift[n].flatten()] ):
            raise ValueError("coding error")
        positions = frac2cart(atoms.get_cell(),folded_positions)
        atoms.positions = positions

    return folded, np.asarray(shift)
