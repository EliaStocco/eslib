import torch
import numpy as np
from ase.io import read
from ase.cell import Cell
from ase import Atoms
from eslib.nn.dataset import make_datapoint
from eslib.nn.dataset import enforce_dependency
from eslib.tools import check_cell_format
from torch_geometric.data import Data
from typing import TypeVar, Tuple

from mace.calculators import MACEliaCalculator

class MaceDipoleBEC():

    model:torch.nn.Module

    def __init__(self,model,device,example): # pbc:bool=None,
        # Load model
        self.mace_calculator = MACEliaCalculator(
            model_path=model, device=device,model_type="AtomicDipolesMACE"
        )
        for param in self.mace_calculator.parameters():
            param.requires_grad = False     

        self.atoms = read(example)
        self.atoms.set_calculator(self.mace_calculator)

    
    def __call__(self,atoms:Atoms):

        self.atoms.set_positions(atoms.get_positions())
        


    
    def get(self,pos=None,cell=None,check=True,X=None)->Tuple[torch.tensor,Data]:
        """Compute the dipole given an the positions and the cell.
        The function requires the atoms to respect the order given by `self._symbols`."""
        # 'cell' has to be in i-PI format (upper triangular) or ase.cell
        if X is None:
            pbc = cell is not None
            cell = self.correct_cell(cell,check=check)
            self.eval()
            requires_grad = {   "pos"        : True,\
                                "lattice"    : True,\
                                "x"          : False,\
                                "edge_vec"   : True,\
                                "edge_index" : False }
            X = self.make_datapoint(lattice=cell.T,positions=pos,requires_grad=requires_grad)
        else:
            # I need to be sure that `X.edge_vec` depends on `X.pos`.
            # I just need to recompute `X.edge_vec` using all the infomation that I already have.
            X = enforce_dependency(X)
        y = self(X)
        if y.shape[0] == 1:
            y = y[0]
        return y, X
    
    def get_jac(self,pos=None,cell=None,y=None,X=None)->Tuple[torch.tensor,torch.tensor]:
        if y is None or X is None:
            y,X = self.get(pos=pos,cell=cell)
        n_batch = len(np.unique(X.batch))
        if n_batch != 1:
            raise ValueError("'get_jac' works only with one batch.")
        N = len(X.pos.flatten())
        jac = torch.full((N,y.shape[0]),torch.nan)
        for n in range(y.shape[0]):
            # y[n].backward(retain_graph=True)
            y[n].backward(retain_graph=True)
            jac[:,n] = X.pos.grad.flatten().detach()
            X.pos.grad.data.zero_()
        return jac,X

    def get_value_and_jac(self,pos=None,cell=None,X=None)->Tuple[torch.tensor,torch.tensor,Data]:
        y,X = self.get(pos=pos,cell=cell,X=X)
        jac,X = self.get_jac(pos=pos,cell=cell,y=y,X=X)
        return y.detach(),jac.detach(),X

    def n_parameters(self:T):
        return sum(p.numel() for p in self.parameters())

