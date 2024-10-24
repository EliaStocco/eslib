from dataclasses import dataclass
from typing import List

from ase import Atoms

from eslib.classes.models.dipole.baseclass import DipoleModel
from eslib.classes.models.mace_model import MACEModel


@dataclass
class DipoleMACECalculator(DipoleModel,MACEModel):
    """A class to handle MACE models predicting dipoles."""
    
    def get(self,traj:List[Atoms],**argv):
        output = self.compute(traj,raw=True,**argv)
        return output["dipole"]
    
    # def summary(self, string="\t"):
    #     DipoleModel.summary(self,string)
    #     MACEModel.summary(self,string)
    
    # def get(self,traj:List[Atoms],**argv):
    #     new_traj = self.compute(traj,**argv)
    #     dipole = AtomicStructures(new_traj).get_info("dipole")
    #     return dipole