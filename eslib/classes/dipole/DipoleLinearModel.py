from dataclasses import dataclass, field
from ase import Atoms
from typing import List
import numpy as np
from eslib.tools import convert
from eslib.physics import compute_dipole_quanta
from copy import copy
from eslib.tools import cart2frac
from eslib.classes.dipole.DipoleModel import DipoleModel

@dataclass
class DipoleLinearModel(DipoleModel):

    ref: Atoms
    bec: np.ndarray
    dipole: np.ndarray
    Natoms: int = field(init=False)  # Natoms is set in __post_init__
    # frame: str = field(default="global")

    def __post_init__(self):
        self.Natoms = self.ref.get_global_number_of_atoms()  # Set Natoms based on the shape of self.ref

        if self.bec is None:
            self.bec = np.full((3*self.Natoms,3),np.nan)
        
        self._check_bec()
        # if self.bec.shape[0] != 3 * self.Natoms:
        #     raise ValueError(f"Invalid shape[0] for 'bec'. Expected {3 * self.Natoms}, got {self.bec.shape}")
        # if self.bec.shape[1] != 3:
        #     raise ValueError(f"Invalid shape[1] for 'bec'. Expected 3, got {self.bec.shape}")

        if self.dipole.shape != (3,):
            raise ValueError(f"Invalid shape for 'dipole'. Expected (3,), got {self.dipole.shape}")

        # if self.frame not in ["global", "eckart"]:
        #     raise ValueError(f"Invalid value for 'frame'. Expected 'global' or 'eckart', got {self.frame}")
    
    def _check_bec(self,bec=None):
        if bec is None:
            bec = self.bec
        if bec.shape[0] != 3 * self.Natoms:
            raise ValueError(f"Invalid shape[0] for 'bec'. Expected {3 * self.Natoms}, got {self.bec.shape}")
        if bec.shape[1] != 3:
            raise ValueError(f"Invalid shape[1] for 'bec'. Expected 3, got {self.bec.shape}")
        return True
        
    def set_bec(self,bec):
        if self._check_bec(bec):
            self.bec = bec

    def get(self,traj:List[Atoms],frame:str="global"):
        """Compute the dipole according to a linear model in the cartesian displacements."""
        # raise ValueError()
        N = len(traj)
        pos = np.zeros((N,self.Natoms,3))
        for n in range(N):
            pos[n,:,:] = traj[n].get_positions()

        # if pos.ndim != 3:
        #     pos = np.atleast_3d(pos)
        #     pos = np.moveaxis(pos, -1, 0)
        if pos[0,:,:].shape != self.ref.positions.shape:
            raise ValueError(f"Invalid shape for 'pos[0]'. Expected {self.ref.positions.shape}, got {pos[0,:,:].shape}")
        out, _ = self._evaluate(pos,frame)
        return out
        
    def _evaluate(self,pos:np.ndarray,frame:str):
        if frame == "eckart" :
            newx, com, rotmat, euler_angles = self.eckart(pos)          
            # compute the model in the Eckart frame
            model, _ = self._evaluate(newx,frame="global")
            # 'rotmat' is supposed to be right-multiplied:
            # vrot = v @ rotmat
            return model, (com, rotmat, euler_angles)

        elif frame == "global" :
            N = len(pos)
            model  = np.full((N,3),np.nan)
            for n in range(N):
                R = pos[n]#.reshape((-1,3))
                delta = np.asarray(R - self.ref.positions)
                delta -= delta.mean(axis=0)
                dD = self.bec.T @ delta.reshape(3*self.Natoms)
                model[n,:] = dD + self.dipole
            return model, (None, None, None)
        
        else:
            raise ValueError(f"Invalid value for 'frame'. Expected 'global' or 'eckart', got {frame}")
            
    def eckart(self,positions:np.ndarray,inplace=False):
        from eslib.classes.eckart import EckartFrame
        from scipy.spatial.transform import Rotation
        m = np.asarray(self.ref.get_masses()) * convert(1,"mass","dalton","atomic_unit")
        eck = EckartFrame(m)
        x    = positions.reshape((-1,self.Natoms,3))
        N    = x.shape[0]
        xref = self.ref.get_positions()
        newx, com, rotmat = eck.align(x,xref)
        # check that everything is okay
        # rotmat = np.asarray([ r.T for r in rotmat ])
        # np.linalg.norm( ( newx - shift ) @ rotmat + shift - x ) 
        euler_angles = np.full((N,3),np.nan)
        for n in range(N):
            # 'rotmat' is supposed to be right multiplied
            # then to get the real rotation matrix we need to 
            # take its transpose
            r =  Rotation.from_matrix(rotmat[n].T)
            angles = r.as_euler("xyz",degrees=True)
            euler_angles[n,:] = angles
        return newx, com, rotmat, euler_angles

    def control_periodicity(self,traj:List[Atoms])->np.ndarray:
        """Returns the indices of the atomic structures that could be 'too far' from the reference configuration"""
        N = len(traj)
        snapshots = np.arange(N).astype(float)
        for n in range(N):
            pos = traj[n].get_positions()
            delta = np.asarray(pos - self.ref.positions)
            # recenter
            frac = cart2frac(self.ref.get_cell(),delta-delta.mean(axis=0))
            if not np.any(abs(frac) > 0.5):
                snapshots[n] = np.nan
        return snapshots[~np.isnan(snapshots)].astype(int)
    
    def get_dipole(self)->np.ndarray:
        return self.dipole
    
    def get_quanta(self)->np.ndarray:
        tmp = copy(self.ref)
        tmp.info["dipole"] = self.dipole
        return np.asarray(compute_dipole_quanta([tmp],in_keyword="dipole")[1][0]).astype(int)

    def get_reference(self):
        return self.ref.copy()
    