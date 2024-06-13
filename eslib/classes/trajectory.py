from copy import deepcopy
from typing import List, Union, TypeVar
import numpy as np
from eslib.tools import convert
from eslib.functional import deprecated
from eslib.classes.aseio import aseio, integer_to_slice_string
T = TypeVar('T', bound='AtomicStructures')

#---------------------------------------#
class AtomicStructures(aseio):
    """
    Class to handle atomic structures with additional functionality:
        - automatic extraction of `info` and `array` from the list of structures
        - methods for getting and setting `info` and `array` attributes
        - methods for converting units of `info` and `array` attributes
        - method for searching if an attribute is present in `info` or `array`
        - method for checking the presence of an attribute in `info`, `array`, or both
        - method for subsampling the AtomicStructures object
    """

    def get_info(self:T,name:str,default:np.ndarray=None)->np.ndarray:
        """
        Get information attribute values for all structures.

        Parameters:
        - name: Name of the information attribute.
        - default: Default value if attribute is missing.

        Returns:
        - np.ndarray: Array of information attribute values.
        """
        output = None
        def set_output(output,n,value):
            if output is None:
                output = np.zeros((len(self),*value.shape))
            output[n] = np.asarray(value)
            return output
           
        for n,structure in enumerate(self):
            if name not in structure.info:
                if default is None:
                    raise ValueError(f'structure n. {n} does not have `{name}` in `info`')
                else:
                    output = set_output(output,n,default)
            else:
                output = set_output(output,n,structure.info[name])
        return output
    
    def get_array(self:T,name:str,default:np.ndarray=None)->np.ndarray:
        """
        Get array attribute values for all structures.

        Parameters:
        - name: Name of the array attribute.
        - default: Default value if attribute is missing.

        Returns:
        - np.ndarray: Array of array attribute values.
        """
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
        """
        Get information or array attribute values for all structures.

        Parameters:
        - name: Name of the attribute.
        - default: Default value if attribute is missing.
        - what: 'info' or 'arrays', if known.

        Returns:
        - np.ndarray: Array of attribute values.
        """
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
        """
        Set information attribute values for all structures (deprecated).

        Parameters:
        - name: Name of the information attribute.
        - data: Array of values to set.
        """
        assert len(self) == data.shape[0]
        for n,atoms in enumerate(self):
            atoms.info[name] = data[n]

    @deprecated(reason="Use `set` instead")
    def set_array(self:T,name:str,data:np.ndarray)->None:
        """
        Set array attribute values for all structures (deprecated).

        Parameters:
        - name: Name of the array attribute.
        - data: Array of values to set.
        """
        assert len(self) == data.shape[0]
        for n,atoms in enumerate(self):
            atoms.arrays[name] = data[n]
        pass    

    def set(self:T,name:str,data:np.ndarray,what:str="unknown")->None:
        """
        Set information or array attribute values for all structures.

        Parameters:
        - name: Name of the attribute.
        - data: Array of values to set.
        - what: 'info' or 'arrays', if known.
        """
        # if what == "unknown":
        #     what = self.search(name)
        # if what == "info" or what == "arrays":
        #     pass
        # else:
        #     raise ValueError("can not find {:s}".format(name))
        assert len(self) == data.shape[0]
        for n,atoms in enumerate(self):
            getattr(atoms,what)[name] = data[n]
        pass

    def convert(self,name,
                family:str=None,
                _from:str="atomic_unit",
                _to:str="atomic_unit",
                inplace:bool=False)->Union[None,np.ndarray]:
        """
        Convert units of attribute values for all structures.

        Parameters:
        - name: Name of the attribute to convert.
        - family: Family of units for conversion.
        - _from: Units to convert from.
        - _to: Units to convert to.
        - inplace: If True, perform conversion in place.

        Returns:
        - np.ndarray: Converted attribute values.
        """
        data = self.get(name)
        data = convert(what=data,family=family,_from=_from,_to=_to)
        if inplace:
            return self.set(name,data)
        else:
            return data
        
    def search(self,name:str)->str:
        """
        Search if attribute is present in 'info' or 'arrays'.

        Parameters:
        - name: Name of the attribute to search.

        Returns:
        - str: 'info', 'arrays', or 'both' if found, 'none' if not found.
        """
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
        """
        Check if attribute is present in 'info', 'arrays', or both.

        Parameters:
        - name: Name of the attribute to check.
        - _all: If True, check if attribute is present in all structures.
        - where: Specify where to search ('info', 'arrays', or None for both).

        Returns:
        - np.ndarray: Boolean array indicating presence of attribute.
        """
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

    # def call(self: T, func) -> np.ndarray:
    #     t = easyvectorize(Atoms)(self)
    #     return t.call(func)
    
    def copy(self:T)->T:
        out = deepcopy(self)
        assert self == out  # Should print True
        assert not (self is out)  # Should print False
        assert all( [ not (a is b) for a,b in zip(self,out) ])
        assert all( [ a == b for a,b in zip(self,out) ])
        return out

def random_water_structure(num_molecules=1):
    from ase import Atoms
    symbols = ['H', 'H', 'O']  # Atomic symbols for water molecule
    water_structure = Atoms()  # Initialize ASE Atoms object
    # Generate random positions for each water molecule
    for _ in range(num_molecules):
        # Randomly generate positions for each atom in the water molecule
        positions = np.random.rand(3, 3)
        # Append the atoms of the water molecule to the overall structure
        water_structure.extend(Atoms(symbols=symbols, positions=positions))
    return water_structure

# @esfmt(None,None)
# def main(args):
#     atoms = random_water_structure(3)
#     structures = AtomicStructures(atoms)
#     pbc = structures.pbc
#     return 

# if __name__ == "__main__":
#     main()