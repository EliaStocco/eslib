from ase import Atoms
from typing import List
from dataclasses import dataclass
import numpy as np
from mace.tools import torch_geometric, torch_tools
from mace.cli.elia_configs import make_dataloader
from mace.modules.models import get_model
from eslib.classes.dipole.DipoleModel import DipoleModel

@dataclass
class DipoleMACECalculator(DipoleModel):
    default_dtype: str
    device: str
    model: str
    model_type: str
    batch_size: int
    charges_key: str

    def __post_init__(self):
        self.initialize()

    def initialize(self):        
        torch_tools.set_default_dtype(self.default_dtype)
        self.device = torch_tools.init_device([self.device])
        self.network = get_model(model_path=self.model,
                               model_type=self.model_type,
                               device=self.device)
        self.network = self.network.to(self.device)  # shouldn't be necessary but seems to help with CUDA problems
        for param in self.network.parameters():
            param.requires_grad = False
        
    def get(self,traj:List[Atoms],**argv):
        self.initialize()
        data_loader:torch_geometric.dataloader.DataLoader = \
            make_dataloader(atoms_list=traj,
                            model=self.network,
                            batch_size=self.batch_size,
                            charges_key=self.charges_key)
        k = 0
        dipoles = np.zeros((len(traj),3))
        for batch in data_loader:
            batch = batch.to(self.device)
            output:dict = self.network(batch.to_dict(), compute_stress=False)
            output = output["dipole"]
            N = len(output)
            dipoles[k:k+N] = torch_tools.to_numpy(output)
            k += N 

        return dipoles