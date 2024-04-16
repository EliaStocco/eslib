from ase import Atoms
from typing import List
from dataclasses import dataclass, field
import numpy as np
from mace.tools import torch_geometric, torch_tools
from mace.cli.elia_configs import make_dataloader
from mace.modules.models import get_model
from eslib.classes.dipole.DipoleModel import DipoleModel
from eslib.classes.mace_model import MACEModel
from eslib.classes.trajectory import AtomicStructures

@dataclass
class DipoleMACECalculator(DipoleModel,MACEModel):
        
    def get(self,traj:List[Atoms],**argv):
        output = self.compute(traj,raw=True,**argv)
        return output["dipole"]
    
    # def get(self,traj:List[Atoms],**argv):
    #     new_traj = self.compute(traj,**argv)
    #     dipole = AtomicStructures(new_traj).get_info("dipole")
    #     return dipole