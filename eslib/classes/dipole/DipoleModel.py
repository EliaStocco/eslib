from ase import Atoms
from typing import List, Dict
from eslib.classes.io import pickleIO
# from abc import ABC, abstractmethod

class DipoleModel(pickleIO):
    # @abstractmethod
    def get(self,traj:List[Atoms],**argv):
        raise NotImplementedError("this method should be overwritten.")
        pass