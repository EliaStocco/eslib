from ase import Atoms
from typing import List
from dataclasses import dataclass
from eslib.classes.models.dipole.DipoleModel import DipoleModel
from classes.models.mace_model import MACEModel

@dataclass
class DipoleMACECalculator(DipoleModel,MACEModel):
    
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