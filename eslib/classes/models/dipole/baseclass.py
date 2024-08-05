import numpy as np
from ase import Atoms
from typing import List
from eslib.classes.models import eslibModel

class DipoleModel(eslibModel):
    def get(self,traj:List[Atoms],**argv)->np.ndarray:
        raise NotImplementedError("this method should be overwritten.")
        
