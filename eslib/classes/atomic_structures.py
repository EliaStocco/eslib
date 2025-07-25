from copy import deepcopy
from typing import Any, List, TypeVar, Union, Tuple, Callable, Optional
from warnings import warn

from ase import Atoms
from ase.cell import Cell
import numpy as np
import pandas as pd
from os import cpu_count

from eslib.classes import Trajectory
from eslib.classes.aseio import aseio, integer_to_slice_string
from eslib.functional import custom_deprecated
from eslib.tools import convert
from eslib.formatting import warning

T = TypeVar('T', bound='AtomicStructures')

#---------------------------------------#
NPROC_DEFAULT = 4
PARALLEL = True
NPROC = min(NPROC_DEFAULT,cpu_count())

#---------------------------------------#
class AtomicStructures(Trajectory,aseio):
    """
    Class to handle atomic structures with additional functionality:
        - automatic extraction of `info` and `array` from the list of structures
        - methods for getting and setting `info` and `array` attributes
        - methods for converting units of `info` and `array` attributes
        - method for searching if an attribute is present in `info` or `array`
        - method for checking the presence of an attribute in `info`, `array`, or both
        - method for subsampling the AtomicStructures object
    """
    
    @classmethod
    def from_atoms(cls,atoms:Atoms,repeat:int=1,clean:bool=False):
        """
        Convert ase.Atoms object to AtomicStructures object.
        """
        if clean:
            from eslib.tools import clean_structure
            atoms = clean_structure(atoms)
        return cls([ atoms.copy() for _ in range(repeat) ])
    
    def num_atoms(self:T):
        n_atoms = self.call(lambda x: x.get_global_number_of_atoms()) # [s.get_global_number_of_atoms() for s in self]
        n_atoms = np.unique(n_atoms)
        if len(n_atoms) > 1:
            raise ValueError("Not all atomic structures have the same number of atoms")
        return n_atoms[0]

    def is_trajectory(self:T)->bool:
        """
        Check if the object is a trajectory.
        """
        # check that all structures have the same number of atoms
        n_atoms = self.call(lambda x: x.get_global_number_of_atoms()) # [s.get_global_number_of_atoms() for s in self]
        n_atoms = np.unique(n_atoms)
        if len(n_atoms) > 1:
            return False
        symbols = self.call(lambda x: x.get_chemical_symbols()) # [s.get_chemical_symbols() for s in self]
        return all([s == symbols[0] for s in symbols])
    
    def get_keys(self:T,what:str="all")->List[str]:
        """
        Get the keys present in all atomic structures, according to the `what` parameter.

        Parameters:
        - `what` (str): Can be 'all', 'info', or 'arrays'. Default is 'all'.

        Returns:
        - List[str]: The keys present in all atomic structures, according to the `what` parameter.

        Raises:
        - ValueError: If `what` is not 'all', 'info', or 'arrays'.
        """

        check_info  = dict()
        check_array = dict()
        
        if what in ["all","info"]:
            keys = self[0].info.keys() 
            if len(keys) > 0:
                for k in keys:
                    for n in range(len(self)):
                        if k not in self[n].info.keys():
                            check_info[k] = False
                            break
                    else:
                        check_info[k] = True

        if what in ["all","arrays"]:
            keys = self[0].arrays.keys()
            if len(keys) > 0:
                for k in keys:
                    for n in range(len(self)):
                        if k not in self[n].arrays.keys():
                            check_array[k] = False
                            break
                    else:
                        check_array[k] = True

        if what not in ["all","info","arrays"]:
            raise ValueError("`what` can be only 'all', 'info', or 'arrays' ")

        check = {**check_info,**check_array}
        
        for key,value in check.items():
            if not value:
                print(f"{warning}: '{key}' is not present in all structrues.")
        # if False in check.values():
        #     raise ValueError("Some checks failed")

        check = {k:v for k,v in check.items() if v is True}

        return list(check.keys())
       

    # ToDo: this might be parallelized with 'multiprocessing'
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
                output = [None]*len(self)
                # try:
                #     output = # np.zeros((len(self),*value.shape))
                # except:
                #     output = # np.zeros(len(self),dtype=type(value))
            # try:
            #     output[n] = np.asarray(value)
            # except:
            output[n] = value
            return output
           
        for n,structure in enumerate(self):
            if name not in structure.info:
                if default is None:
                    raise ValueError(f'structure n. {n} does not have `{name}` in `info`')
                else:
                    output = set_output(output,n,default)
            else:
                output = set_output(output,n,structure.info[name])
        return np.asarray(output)
    
    # ToDo: this might be parallelized with 'multiprocessing'
    def get_array(self:T,name:str,default:np.ndarray=None)->Union[np.ndarray,List[np.ndarray]]:
        """
        Get array attribute values for all structures.

        Parameters:
        - name: Name of the array attribute.
        - default: Default value if attribute is missing.

        Returns:
        - np.ndarray: Array of array attribute values.
        """
        # output = None
        # def set_output(output,n,value):
        #     if output is None:
        #         output = np.zeros((len(self),*value.shape))
        #     try:
        #         output[n] = np.asarray(value)
        #     except:
        #         output[n] = value
        #     return output
            
        output = [None]*len(self)
        # shapes = [None]*len(self)
        for n,structure in enumerate(self):
            if name not in structure.arrays:
                if default is None:
                    raise ValueError(f"structure n. {n} does not have '{name}' in `arrays`")
                else:
                    output[n] = default
                    # shapes[n] = default.shape
            else:
                # value = structure.arrays[name]
                output[n] = structure.arrays[name]
                # shapes[n] = value.shape
        try:
            return np.asarray(output)
        except:
            return output

    def get(self:T,name:str,default:np.ndarray=None,what:str="all")->np.ndarray:
        """
        Get information or array attribute values for all structures.

        Parameters:
        - name: Name of the attribute.
        - default: Default value if attribute is missing.
        - what: 'info' or 'arrays', if known.

        Returns:
        - np.ndarray: Array of attribute values.
        """
        if what in ["unknown","all"]:
            what = self.search(name)
        if what == "info":
            return self.get_info(name,default)
        elif what == "arrays":
            return self.get_array(name,default)
        else:
            message = "Can not find {:s}.\nAvailable infos: {:s}\nAvailable arrays: {:s}".format(
                    name, ', '.join(self[0].info.keys()), ', '.join(self[0].arrays.keys()))
            raise ValueError(message)
        
    def overview(self: T, string: str = "\t") -> str:
        """
        Print overview of the AtomicStructures object.

        Returns:
            str: Overview of the AtomicStructures object.
        """
        infos = ', '.join(self.get_keys("info"))
        arrays = ', '.join(self.get_keys("arrays"))
        message = (
            "\n" 
            "Available infos: [{infos}]\n"
            "Available arrays: [{arrays}]"
        ).format(infos=infos, arrays=arrays)
        message = message.replace("\n", f"\n{string}")
        print(message)

    # ToDo: this should be generalized to handle structures with different number of atoms
    def summary(self: T) -> pd.DataFrame:
        """
        Create a pandas DataFrame summarizing the content of the AtomicStructures object.

        The DataFrame has columns:
        - key: Name of the attribute.
        - i/a: 'info' or 'arrays', depending on the type of attribute.
        - numeric: Boolean indicating if the attribute is numeric.
        - dtype: Data type of the attribute.
        - shape: Shape of the attribute, if numeric.

        Returns:
            pd.DataFrame: DataFrame with the summary.
        """
        df = pd.DataFrame(columns=["key","i/a","numeric","dtype","shape"])
        df["key"] = self.get_keys()
        info = self.get_keys("info")
        array = self.get_keys("arrays")
        
        is_traj = self.is_trajectory()

        # Iterate over the columns
        for n,key in enumerate(df["key"]):
            if key == "original-file":
                continue
            # Determine if it is an info or an array attribute
            if key in info:
                df.at[n,"i/a"] = "info"
            elif key in array:
                df.at[n,"i/a"] = "array"
            else:
                raise ValueError("Unknown attribute type")

            # Get the value of the attribute
            what = self.search(key)
            if what == "arrays" and not is_traj:
                continue
            value = self.get(key)
            try:
                # If the value is numeric, get its dtype and shape
                df.at[n,"dtype"] = value.dtype
                df.at[n,"numeric"] = np.issubdtype(value.dtype, np.number)
                df.at[n,"shape"] = value.shape
            except:
                # Otherwise, get its type and set shape to None
                df.at[n,"dtype"] = type(value[0])
                df.at[n,"numeric"] = False
                df.at[n,"shape"] = None

        return df

    # ToDo: this might be parallelized with 'multiprocessing'
    @custom_deprecated(reason="Use `set` instead")
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

    # ToDo: this might be parallelized with 'multiprocessing'
    @custom_deprecated(reason="Use `set` instead")
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

    # ToDo: this might be parallelized with 'multiprocessing'
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
        if what == "unknown":
            what = self.search(name)
            if what not in ["info","arrays"]:
                raise ValueError("can not find {:s}".format(name))
        for n,atoms in enumerate(self):
            getattr(atoms,what)[name] = data[n]
            # if what not in ["positions"]:
            #     getattr(atoms,what)[name] = data[n]
            # elif what == "positions":
            #     atoms.set_positions(data[n])
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
        what = self.search(name)
        if inplace:
            return self.set(name,data,what=what)
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

        if info and arrays:         return "both"
        if not info and not arrays: return "none"
        if info:                    return "info"
        if arrays:                  return "arrays"

    def has(self:T,name:str,_all:bool=True,where:str=None)->np.ndarray:
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
    
    def __getitem__(self, item)->T:
        result = super().__getitem__(item)
        return AtomicStructures(result) if isinstance(result, list) else result
    
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
    
    def extract_random(self:T,N:int) -> Tuple[np.ndarray,T]:
        import random
        indices = np.arange(len(self))
        random.shuffle(indices)
        indices = indices[:N]
        assert len(indices) == N, "coding error"
        return indices,self.subsample(indices)
    
    def remove(self:T,indices) ->T :
        assert len(indices) == len(set(indices)), "The indices are not unique."
        filtered_structures = [item for i, item in enumerate(self) if i not in indices]
        return AtomicStructures(filtered_structures)

    def remove_info_array(self:T,key:str,what:str=None)->T:
        if what is None:
            what = self.search(key)
        if what == "info":
            for structure in self:
                if key in structure.info:
                    del structure.info[key]
        elif what == "arrays":
            for structure in self:
                if key in structure.arrays:
                    del structure.arrays[key]
        else:
            raise ValueError(f"Unknown attribute '{key}' in {what}.")

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
    
    def info2pandas(self:T,discard=[],include=None)->pd.DataFrame:
        """
        Convert the info attributes of the AtomicStructures object to a pandas DataFrame.

        Parameters:
        - discard (list): List of keys to discard from the DataFrame. Defaults to an empty list.
        - include (list): List of keys to include in the DataFrame. Defaults to None, which means all keys are included.

        Returns:
        - pd.DataFrame: A pandas DataFrame containing the info attributes of the AtomicStructures object.
        """
        df = self.summary()
        df = df[df["i/a"]=="info"]
        df.set_index("key",inplace=True)
        if include is not None:
            df = df[df.index.isin(include)]        
        N = len(self)
        output = pd.DataFrame(index=np.arange(N))
        
        for key,row in df.iterrows():
            if key in discard:
                continue
            data = self.get(key)
            if data.ndim == 1:
                output[key] = data
            elif data.ndim == 2:
                for n in range(data.shape[1]):
                    output[f"{key}_{n}"] = data[:,n]
            else:
                warn (f"Skipping {key} with shape {row['shape']}")
        return output
        
    # def fold(self:T):
    #     pos = self.get("positions")
    #     cells = self.get_cells()
    #     pos = pos % 1 
    #     for n,_ in enumerate(self):
    #         self[n].set_scaled_positions(pos[n])
    
    def get_cells(self:T,as_numpy:bool=False)->List[Cell]:
        """Return a list of ase.cell.Cells"""
        if as_numpy:
            cells = self.get_cells(as_numpy=False)
            cells = np.asarray([np.asarray(c).T for c in cells])
            return cells
        else:
            return [a.get_cell() for a in self]
    
    def get_chemical_symbols(self:T,unique:bool=False):
        """Return a list of chemical symbols"""
        if unique:
            species = [ np.unique(a.get_chemical_symbols()) for a in self]
            species = np.asarray(species).flatten()
            return np.unique(species).tolist()
        else:
            return [a.get_chemical_symbols() for a in self]
        
    # ToDo: this might be parallelized with 'multiprocessing'
    def get_scaled_positions(self:T)->Union[List[np.ndarray],np.ndarray]:
        """Return a list of scaled positions"""
        if self.is_trajectory():
            out = np.zeros((len(self),self.num_atoms(),3))
            for n,a in enumerate(self):
                out[n,:,:] = a.get_scaled_positions()
            return out
        else:
            out = [None]*len(self)
            for n,a in enumerate(self):
                out[n] = a.get_scaled_positions()
            return out
        
    def unwrap(self:T,inplace:bool=True)->np.ndarray:
        """Unwrap the positions of the atoms in the structures"""
        assert self.is_trajectory(), "The atomic structures should all have the same number of atoms."
        frac = self.get_scaled_positions()
        frac = frac % 1
        assert frac.shape == (len(self),self.num_atoms(),3)
        new_frac = np.unwrap(frac,axis=0,period=1)
        cells = self.get_cells(as_numpy=True)
        pos = np.einsum("sij,saj->sai", cells, new_frac) # s: structure, a: atom, i,j: xyz
        if inplace:
            self.set("positions",pos)
        return pos
          
    def apply(self, func: Callable[[Atoms], None], parallel: bool = None, processes: Optional[int] = None):
        """
        Apply a function to each Atoms object.

        Parameters:
            func: function to apply to each Atoms
            parallel: run in parallel if True
            processes: number of worker processes (default: number of CPUs)

        Returns:
            None
        """
        if parallel is None:
            parallel = PARALLEL
        if parallel:
            from multiprocessing import Pool
            if processes is None:
                processes = NPROC
            with Pool(processes=processes) as pool:
                pool.map(func, [ a for a in self])
        else:
            for structure in self:
                structure = func(structure)
        return

    def call(self, func: Callable[[Atoms], List[Any]], parallel: bool = None, processes: Optional[int] = None)->List[Any]:
        """
        Apply a function to each Atoms object.

        Parameters:
            func: function to apply to each Atoms (can be lambda)
            parallel: run in parallel if True
            processes: number of threads (default: number of CPUs)

        Returns:
            List of results from applying func to each Atoms
        """
        structures = list(self)
        
        if parallel is None:
            parallel = PARALLEL

        if not parallel:
            return [func(a) for a in structures]

        if processes is None:
            processes = NPROC
        
        from concurrent.futures import ThreadPoolExecutor
        from itertools import chain
        from math import ceil

        # Split into chunks ~1 per thread
        chunk_size = ceil(len(structures) / processes)
        chunks = [structures[i:i + chunk_size] for i in range(0, len(structures), chunk_size)]

        def batch_worker(batch):
            return [func(a) for a in batch]

        with ThreadPoolExecutor(max_workers=processes) as executor:
            results = executor.map(batch_worker, chunks)

        return list(chain.from_iterable(results))

    @staticmethod
    def set_parallel(parallel: bool = True, nproc: Optional[int] = None):
        """
        Set the parallelization mode and number of processes for AtomicStructures.

        Parameters:
            parallel: If True, run in parallel.
            nproc: Number of worker processes (default: min among 4 and number of CPUs).
        """
        global PARALLEL, NPROC
        PARALLEL = parallel
        if nproc is None:
            NPROC = min(NPROC_DEFAULT,cpu_count())
        else:
            NPROC = nproc
    
#---------------------------------------#
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