from typing import List

import numpy as np
from ase import Atoms

from eslib.classes.models import eslibModel


class DipoleModel(eslibModel):
    """A class to handle models predicting dipoles."""

    # @abc.abstractmethod
    def get(self,traj:List[Atoms],**argv)->np.ndarray:
        """Return the dipole for a list of Atoms objects."""
        raise NotImplementedError("this method should be overwritten.")
        
