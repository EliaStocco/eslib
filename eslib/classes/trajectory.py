from ase.io import read, write, string2index
from ase import Atoms
# from .vectorize import easyvectorize
from io import TextIOWrapper
import re
# import ipi.utils.mathtools as mt
import math
import numpy as np
# from deprecated import deprecated
from eslib.tools import convert
from eslib.functional import deprecated
from eslib.classes.io import pickleIO
from eslib.classes.aseio import aseio, integer_to_slice_string
from typing import List, Union, TypeVar

T = TypeVar('T', bound='AtomicStructures')

#---------------------------------------#
class AtomicStructures(aseio):
    """Class to handle atomic structures:
        - automatic extraction of `info` and `array` from the list of structures
    """
    
    def get_info(self:T,name:str,default:np.ndarray=None)->np.ndarray:
        output = None
        def set_output(output,n,value):
            if output is None:
                output = np.zeros((len(self),*value.shape))
            output[n] = np.asarray(value)
            return output
            
        for n,structure in enumerate(self):
            if name not in structure.info:
                if default is None:
                    raise ValueError("structure n. {:n} does not have '{:s}' in `info`".format(n,name))
                else:
                    output = set_output(output,n,default)
            else:
                output = set_output(output,n,structure.info[name])
        return output
    
    def get_array(self:T,name:str,default:np.ndarray=None)->np.ndarray:
        output = None
        def set_output(output,n,value):
            if output is None:
                output = np.zeros((len(self),*value.shape))
            output[n] = np.asarray(value)
            return output
            
        for n,structure in enumerate(self):
            if name not in structure.arrays:
                if default is None:
                    raise ValueError("structure n. {:n} does not have '{:s}' in `arrays`".format(n,name))
                else:
                    output = set_output(output,n,default)
            else:
                output = set_output(output,n,structure.arrays[name])
        return output

    def get(self:T,name:str,default:np.ndarray=None,what:str="unknown")->np.ndarray:
        if what == "unknown":
            what = self.search(name)
        if what == "info":
            return self.get_info(name,default)
        elif what == "arrays":
            return self.get_array(name,default)
        else:
            raise ValueError("can not find {:s}".format(name))

    @deprecated(reason="Use `set` instead")
    def set_info(self:T,name:str,data:np.ndarray)->None:
        assert len(self) == data.shape[0]
        for n,atoms in enumerate(self):
            atoms.info[name] = data[n]
        pass
    
    @deprecated(reason="Use `set` instead")
    def set_array(self:T,name:str,data:np.ndarray)->None:
        assert len(self) == data.shape[0]
        for n,atoms in enumerate(self):
            atoms.arrays[name] = data[n]
        pass    

    def set(self:T,name:str,data:np.ndarray,what:str="unknown")->None:
        # if what == "unknown":
        #     what = self.search(name)
        if what == "info" or what == "arrays":
            pass
        else:
            raise ValueError("can not find {:s}".format(name))
        assert len(self) == data.shape[0]
        for n,atoms in enumerate(self):
            getattr(atoms,what)[name] = data[n]
        pass

    def convert(self,name,
                family:str=None,
                _from:str="atomic_unit",
                _to:str="atomic_unit",
                inplace:bool=False)->Union[None,np.ndarray]:
        data = self.get(name)
        data = convert(what=data,family=family,_from=_from,_to=_to)
        if inplace:
            return self.set(name,data)
        else:
            return data
        
    def search(self,name:str)->str:

        # is it in `info`?
        booleans = [ name in s.info for s in self ]
        info = np.all(booleans)

        # is it in `arrays`?
        booleans = [ name in s.arrays for s in self ]
        arrays = np.all(booleans)

        if info and arrays: "both"
        if not info and not arrays: return "none"
        if info: return "info"
        if arrays: return "arrays"

    def is_there(self:T,name:str,_all:bool=True,where:str=None)->np.ndarray:
        if where is None:
            booleans = [ name in s.info or name in s.arrays for s in self ]
        elif where in ["i","info"]:
            booleans = [ name in s.info for s in self ]
        elif where in ["a","array","arrays"]:
            booleans = [ name in s.arrays for s in self ]
        else:
            raise ValueError("`where` can be only None, ['i', 'info'], or ['a', 'array', 'arrays'] ")
        return np.all(booleans) if _all else np.any(booleans)
    
    def subsample(self:T, indices: List[int]) -> T:
        """
        Subsample the AtomicStructures object using the provided indices.

        Parameters:
        - indices: A list of integers specifying the indices to keep.

        Returns:
        - AtomicStructures: A new AtomicStructures object containing the subsampled structures.
        """
        if isinstance(indices,list) or isinstance(indices,np.ndarray):
            subsampled_structures = AtomicStructures([self[i] for i in indices])
        else:
            indices = integer_to_slice_string(indices)
            subsampled_structures = self[indices]
            subsampled_structures = AtomicStructures(subsampled_structures)
        return subsampled_structures
    