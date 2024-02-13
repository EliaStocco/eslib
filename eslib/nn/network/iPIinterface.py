import torch
import numpy as np
from ase.io import read
from eslib.nn.dataset import make_datapoint
from torch_geometric.data import Data
from abc import ABC, abstractproperty, abstractmethod
from typing import TypeVar, Tuple
T = TypeVar('T', bound='iPIinterface')

class iPIinterface(ABC):

    @abstractproperty
    def default_dtype(self:T): 
        pass

    @abstractmethod
    def eval(self:T)->T:
        pass

    def __init__(self:T,max_radius:float,**kwargs): # pbc:bool=None,
        # super().__init__(max_radius=max_radius)
        self._max_radius = max_radius
        #self._pbc = pbc
        self._symbols = None        

    def make_datapoint(self,lattice, positions,**argv)->Data:
        other = { "lattice":lattice,
                  "positions":positions,
                  "symbols":self._symbols,
                  "max_radius":self._max_radius,
                  "default_dtype": self.default_dtype,
                  "pbc": lattice is not None and np.linalg.det(lattice) != np.inf }
        return make_datapoint(**other,**argv) 
        
    def store_chemical_species(self,file=None,atoms=None,**argv):
        if file is not None:
            atoms = read(file,**argv)
        elif atoms is None:
            raise ValueError("'file' or 'atoms' can not be both None.")
        self._symbols = atoms.get_chemical_symbols()
    
    def correct_cell(self,cell=None):
        if cell is None:
            cell = torch.eye(3).fill_diagonal_(torch.inf)
        return cell
    
    def get(self,pos,cell=None)->Tuple[torch.tensor,Data]:
        # 'cell' has to be in i-PI format
        pbc = cell is not None
        cell = self.correct_cell(cell)
        self.eval()
        requires_grad = {   "pos"        : True,\
                            "lattice"    : True,\
                            "x"          : False,\
                            "edge_vec"   : True,\
                            "edge_index" : False }
        X = self.make_datapoint(lattice=cell.T,positions=pos,requires_grad=requires_grad)
        y = self(X)[0]
        return y, X
    
    def get_jac(self,pos,cell=None,y=None,X=None)->Tuple[torch.tensor,torch.tensor]:
        if y is None or X is None:
            y,X = self.get(pos=pos,cell=cell)
        N = len(X.pos.flatten())
        jac = torch.full((N,y.shape[0]),torch.nan)
        for n in range(y.shape[0]):
            y[n].backward(retain_graph=True)
            jac[:,n] = X.pos.grad.flatten().detach()
            X.pos.grad.data.zero_()
        return jac,X

    def get_value_and_jac(self,pos,cell=None)->Tuple[torch.tensor,torch.tensor,Data]:
        y,X = self.get(pos,cell)
        jac,X = self.get_jac(pos,cell,y=y,X=X)
        return y.detach(),jac.detach(),X

    def n_parameters(self:T):
        return sum(p.numel() for p in self.parameters())

