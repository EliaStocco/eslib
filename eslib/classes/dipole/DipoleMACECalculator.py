from ase import Atoms
from typing import List
from dataclasses import dataclass
from eslib.classes.dipole.DipoleModel import DipoleModel
from eslib.classes.mace_model import MACEModel

@dataclass
class DipoleMACECalculator(DipoleModel,MACEModel):
    
    def get(self,traj:List[Atoms],**argv):
        output = self.compute(traj,raw=True,**argv)
        return output["dipole"]
    
    def summary(self, string="\t"):
        super().summary(string)
    
    # def get(self,traj:List[Atoms],**argv):
    #     new_traj = self.compute(traj,**argv)
    #     dipole = AtomicStructures(new_traj).get_info("dipole")
    #     return dipole