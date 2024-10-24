from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, TypeVar

import numpy as np
import torch
from ase import Atoms
from ase.cell import Cell
from ase.io import read
from torch_geometric.data import Data

from eslib.nn.dataset import enforce_dependency, make_datapoint
from eslib.tools import check_cell_format

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
        
    def store_chemical_species(self,x,**argv):
        """Save the chemical symbols of an atomic structure into an object attribute. 
        You can provide a file (whoch will be read using `ase.io.read`), an `ase.Atoms` object or a `list` of `str` with the symbols."""
        if isinstance(x,str):
            atoms = read(x,**argv)
            self.store_chemical_species(atoms)
        elif isinstance(x,Atoms):
            symbols = x.get_chemical_symbols()
            self.store_chemical_species(symbols)
        elif isinstance(x,list):
            if not all(isinstance(elem, str) for elem in x):
                raise TypeError("the provided list should contain 'str' elements.")
            self._symbols = x
        else:
            raise TypeError("type not implemented")
    
    def correct_cell(self,cell=None,check:bool=True):
        """Set the cell to a cube with infinite lattice vectors or check that it is upper triangular."""
        if cell is None:
            cell = torch.eye(3).fill_diagonal_(torch.inf)
        if isinstance(cell,Cell):
            cell = np.asarray(cell).T
        if check and not check_cell_format(cell):
            raise ValueError("provided cell has a wrong format.")
        return cell
    
    def get_from_structure(self,atoms:Atoms)->Tuple[torch.tensor,Data]:
        """Compute the dipole given an atomic structure as an `ase.Atoms` object.
        The function takes care of an eventual permutation of the atoms.
        """
        pos = atoms.get_positions()
        cell = atoms.get_cell() if np.all(atoms.get_pbc()) else None
        symbols = self._symbols
        self._symbols = atoms.get_chemical_symbols()
        y = self.get(pos=pos,cell=cell)
        self._symbols =symbols
        return y

    
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

    # def get_jac(self, pos=None, cell=None, y=None, X=None) -> Tuple[torch.tensor, torch.tensor]:    
    #     if y is None or X is None:
    #         y, X = self.get(pos=pos, cell=cell)
        
    #     num_batches = y.shape[0]
    #     N = len(X.pos.flatten())/num_batches
    #     jac = torch.full((num_batches,N,3), torch.nan)  # Initialize the Jacobian matrix
        
    #     # Loop over each batch dimension
    #     for i in range(num_batches):
    #         y_batch = y[i]  # Get the i-th batch of y
    #         # y_batch.backward(retain_graph=True)  # Backpropagate
    #         # jac[:, i] = X.pos.grad.flatten().detach()  # Store the gradients in the Jacobian matrix
    #         # X.pos.grad.data.zero_()  # Zero out the gradients for the next iteration
    #         for n in range(3):
    #             y_batch[n].backward(retain_graph=True)
    #             jac[:,n] = X[i,n].pos.grad.flatten().detach()
    #             X.pos.grad.data.zero_()
        
    #     return jac, X

    def get_value_and_jac(self,pos=None,cell=None,X=None)->Tuple[torch.tensor,torch.tensor,Data]:
        y,X = self.get(pos=pos,cell=cell,X=X)
        jac,X = self.get_jac(pos=pos,cell=cell,y=y,X=X)
        return y.detach(),jac.detach(),X

    def n_parameters(self:T):
        return sum(p.numel() for p in self.parameters())

