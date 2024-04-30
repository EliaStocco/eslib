import xarray as xr
import numpy as np
from .io import pickleIO
from typing import Union
from eslib.formatting import float_format
import warnings
# Filter out the warning by category and message
warnings.filterwarnings("ignore", category=FutureWarning, message="xarray subclass bec should explicitly define __slots__")

class bec(xr.DataArray,pickleIO):
    __slots__ = ('_data', '_dtype', '_file', '_other_attribute')  # Add any additional attributes you may have

    @classmethod
    def from_extxyz(cls,file:str,name:str="bec"):
        from .trajectory import AtomicStructures, array
        atoms = AtomicStructures.from_file(file)
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
    
    def to_file(self,file:str):
        if file.endswith("pickle"):
            self.to_pickle(file)
        elif file.endswith("txt"):
            with open(file,"w") as ffile:
                for n,Z in enumerate(self):
                    x = np.asarray(Z)
                    header = "structure {:d}".format(n)
                    np.savetxt(ffile, x, fmt=float_format,header=header)
        else:
            raise ValueError("not implemented yet for file '{:s}'".format(file))
        
    def summary(self):
        structure_mean = self.mean("structure")
        # structure_atoms_mean = structure_mean
        output = {
            "structure-mean" : structure_mean.to_numpy().tolist(),
            # "structure-atoms-mean" : list(structure_atoms_mean),
        }
        return output

    @classmethod
    def from_components(cls,Zx:np.ndarray,Zy:np.ndarray,Zz:np.ndarray):
        if Zx.shape != Zy.shape or Zx.shape != Zz.shape:
            raise ValueError("Zx, Zy, and Zz must have the same shape")
        Nstructures = Zx.shape[0]
        Ndof = 3*Zx.shape[1]
        array = np.zeros((Nstructures,Ndof,3))
        #
        Zx = Zx.reshape((Nstructures,-1))
        Zy = Zy.reshape((Nstructures,-1))
        Zz = Zz.reshape((Nstructures,-1))
        # 
        array[:,:,0] = Zx
        array[:,:,1] = Zy
        array[:,:,2] = Zz
        #
        obj = xr.DataArray(array.copy(), dims=('structure', 'dof', 'dir'))
        return cls(obj)

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
        return int(self.shape[1]/3)

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
        tmp = self.expand_with_atoms()
        if index is not None:
            array = tmp.isel(structure=index)
            return np.allclose(array.sum(dim='atom'),np.zeros((3,3)))
        else:
            return np.asarray([ tmp.isel(structure=i) for i in range(tmp.shape[0]) ])
    
    # def check_asr(self,index:int=None):
    #     """Check whether a specific BEC satisfies the Acoustic Sum Rule."""
    #     if index is not None:
    #         array = self.isel(structure=index)
    #         return np.allclose(array.sum(dim='dof'),np.zeros(3))
    #     else:
    #         return np.asarray([ self.isel(structure=i) for i in range(self.shape[0]) ])
        
    def force_asr(self,index:int=None):
        """Enforce the Acoustic Sum Rule."""
        mean = self.mean(dim="dof")
        self = self-mean
        return mean
    
    def expand_with_atoms(self):
        array = self.to_numpy()
        array = array.reshape((array.shape[0],int(array.shape[1]/3),3,3))
        obj = xr.DataArray(array.copy(), dims=('structure', 'atom', 'xyz', 'dir'))
        return bec(obj)