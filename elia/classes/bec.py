import xarray as xr
import numpy as np
from .io import pickleIO
from typing import Union
class bec(xr.DataArray,pickleIO):
    __slots__ = ('_data', '_dtype', '_file', '_other_attribute')  # Add any additional attributes you may have

    @classmethod
    def from_extxyz(cls,file:str,name:str="bec"):
        from .trajectory import trajectory, array
        atoms = trajectory(file)
        becs = array(atoms,name)
        return cls.from_numpy(becs)

    @classmethod
    def from_file(cls,file:str,natoms:int):
        if file.endswith("txt"):
            array = np.loadtxt(file)
        elif file.endswith("npy"):
            array = np.load(file)
        else:
            try:
                array = np.load(file)
            except:
                array = np.load(file)
        array = array.reshape((-1,3*natoms,3))
        return cls.from_numpy(array)

    @classmethod
    def from_numpy(cls,array:np.ndarray):
        """Create a 'bec' object from a numpy ndarray."""

        # array = np.atleast_3d(array)

        if len(array.shape) != 3:
            # raise ValueError("only 3D array are supported")
            array = array.reshape((1,*array.shape))
        
        Nstruc = array.shape[0]
        if array.shape[2] == 9:
            Natoms = array.shape[1]
            empty = np.full((Nstruc,3*Natoms,3),np.nan)
            for s in range(empty.shape[0]): # cycle over the atomic structures
                empty[s,:,:] = array[s,:,:].reshape((3*Natoms,3))
            array = empty.copy()
        elif array.shape[2] == 3:
            pass
        else:
            raise ValueError("wrong number of columns")
        
        obj = xr.DataArray(array.copy(), dims=('structure', 'dof', 'dir'))
        return cls(obj)

    @property
    def natoms(self)->int:
        return self.shape[1]/3

    def norm2(self,reference:Union[xr.DataArray,np.ndarray]):
        """Computes the norm squared per atom of the difference w.r.t. a reference BEC."""
        if not isinstance(reference,xr.DataArray):
            reference = xr.DataArray(reference,dims=('dof','dir'))
        delta = self - reference
        delta = np.square(delta)
        delta = delta.sum(dim=('dof','dir'))
        delta = delta/self.natoms
        # delta = np.sqrt(delta)
        return delta
    
    def check_asr(self,index:int=None):
        """Check whether a specific BEC satisfies the Acoustic Sum Rule."""
        if index is not None:
            array = self.isel(structure=index)
            return np.allclose(array.sum(dim='dof'),np.zeros(3))
        else:
            return np.asarray([ self.isel(structure=i) for i in range(self.shape[0]) ])
        
    def force_asr(self,index:int=None):
        """Enforce the Acoustic Sum Rule."""
        mean = self.mean(dim="dof")
        self = self-mean
        return mean