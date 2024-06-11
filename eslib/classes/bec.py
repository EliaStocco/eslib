import xarray as xr
import numpy as np
from .io import pickleIO
from typing import Union, List, Dict, Tuple, TypeVar, Type
from eslib.formatting import float_format
import warnings
# Filter out the warning by category and message
warnings.filterwarnings("ignore", category=FutureWarning, message="xarray subclass bec should explicitly define __slots__")

T = TypeVar('T', bound='bec')

class bec(xr.DataArray,pickleIO):
    # __slots__ = xr.DataArray.__slots__ #('_data', '_dtype', '_file', '_other_attribute')  # Add any additional attributes you may have

    # @classmethod
    # def from_extxyz(cls,file:str,name:str="bec"):
    #     from .trajectory import AtomicStructures, array
    #     atoms = AtomicStructures.from_file(file)
    #     becs = array(atoms,name)
    #     return cls.from_numpy(becs)

    @classmethod
    def from_file(cls:Type[T],file:str,natoms:int=None)->T:
        if file.endswith("txt"):
            array = np.loadtxt(file)
        elif file.endswith("npy"):
            array = np.load(file)
        else:
            try:
                array = np.load(file)
            except:
                array = np.load(file)
        if natoms is None:
            natoms = int(array.shape[0]/3)
        array = array.reshape((-1,3*natoms,3))
        return cls.from_numpy(array)
    
    def to_file(self:T,file:str)->None:
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
        
    def summary(self:T,symbols:List[str]=None):
        structure_mean = self.mean("structure")
        structure_std = self.std("structure")
        structure_rel = 100*structure_std/structure_mean
        trace_atoms, _ = self.trace("atoms")
        if symbols is not None:
            trace_species, std_species = self.trace(average="species",symbols=symbols,trace_atoms=trace_atoms)
            traces = trace_species.mean("structure")
            traces = dict(zip(np.unique(symbols).tolist(), traces.to_numpy().tolist()))
            std = std_species.mean("structure")
            std = dict(zip(np.unique(symbols).tolist(), std.to_numpy().tolist()))
        # structure_atoms_mean = structure_mean
        output = {
            "species-mean" : traces,
            "species-std" : std,
            "structure-percentage" : structure_rel.to_numpy().tolist(),  
            "structure-std" : structure_std.to_numpy().tolist(),  
            "structure-mean" : structure_mean.to_numpy().tolist(),            
        }
        # for k,value in output.items():
        #     output[k] = value.to_numpy().tolist()
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

    def norm2(self:T,reference:Union[xr.DataArray,np.ndarray]):
        """Computes the norm squared per atom of the difference w.r.t. a reference BEC."""
        if not isinstance(reference,xr.DataArray):
            reference = xr.DataArray(reference,dims=('dof','dir'))
        delta = self - reference
        delta = np.square(delta)
        delta = delta.sum(dim=('dof','dir'))
        delta = delta/self.natoms
        # delta = np.sqrt(delta)
        return delta
    
    def check_asr(self:T,index:int=None):
        """Check whether a specific BEC satisfies the Acoustic Sum Rule."""
        tmp = self.expand_with_atoms()
        if index is not None:
            array = tmp.isel(structure=index)
            return np.allclose(array.sum(dim='atom'),np.zeros((3,3)))
        else:
            return np.asarray([ tmp.isel(structure=i) for i in range(tmp.shape[0]) ])
    
    # def check_asr(self:T,index:int=None):
    #     """Check whether a specific BEC satisfies the Acoustic Sum Rule."""
    #     if index is not None:
    #         array = self.isel(structure=index)
    #         return np.allclose(array.sum(dim='dof'),np.zeros(3))
    #     else:
    #         return np.asarray([ self.isel(structure=i) for i in range(self.shape[0]) ])
        
    def force_asr(self:T,index:int=None):
        """Enforce the Acoustic Sum Rule."""
        mean = self.mean(dim="dof")
        self = self-mean
        return mean
    
    def expand_with_atoms(self):
        array = self.to_numpy()
        array = array.reshape((array.shape[0],int(array.shape[1]/3),3,3))
        obj = xr.DataArray(array.copy(), dims=('structure', 'atom', 'xyz', 'dir'))
        return bec(obj)
    
    def trace(self:T,average:str="atoms",symbols:List[str]=None,trace_atoms:xr.DataArray=None)->Tuple[xr.DataArray,xr.DataArray]:
        symbols = np.asarray(symbols) 
        if average == "atoms":
            atomic_bec = self.expand_with_atoms()
            trace = xr.DataArray(np.zeros(atomic_bec.shape[0:2]), dims=('structure', 'atom'))
            # std = trace.copy()
            for i,s in enumerate(atomic_bec): # cycle over structures
                for j,a in enumerate(s): # cycle over atoms
                    # val = np.trace(a)/3
                    z = np.diagonal(a)
                    # assert val == np.mean(z)
                    trace[i,j] = np.mean(z) # trace(Z)/3
                    # std[i,j] = np.std(z)
            return trace, None
        elif average == "species":
            if trace_atoms is None:
                trace_atoms = self.trace(average="atoms")
            species,index = np.unique(symbols,return_inverse=True)
            assert all( [ a== b for a,b in zip(species[index],symbols)])
            trace = xr.DataArray(np.zeros((trace_atoms.shape[0],len(species))), dims=('structure', 'species'))
            std = trace.copy()
            for n,s in enumerate(species): # cycle over species
                ii = np.where(symbols == s)[0]
                trace[:,n] = trace_atoms[:,ii].mean("atom")
                std[:,n] = trace_atoms[:,ii].std("atom")
            return trace, std
        else:
            raise ValueError("coding error")
        
    def relative_to(self:T,arr:T)->T:
        trace,_ = arr.trace("atoms")
        trace = np.repeat(trace.values, 9).reshape(trace.shape[0], trace.shape[1] * 3, 3)
        return self/trace


