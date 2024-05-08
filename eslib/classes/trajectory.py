from ase.io import read, write, string2index
from ase import Atoms
from .vectorize import easyvectorize
from eslib.functions import read_comments_xyz
import re
# import ipi.utils.mathtools as mt
import math
import numpy as np
# from deprecated import deprecated
from eslib.tools import convert
from eslib.functional import deprecated
from eslib.classes.io import pickleIO
from typing import List, Union, TypeVar

T = TypeVar('T', bound='AtomicStructures')

AStype = List[Atoms]

deg2rad     = np.pi / 180.0
abcABC      = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
abcABCunits = re.compile(r'\{([^}]+)\}')
step        = re.compile(r"Step:\s+(\d+)")

def abc2h(a, b, c, alpha, beta, gamma):
    """Returns a lattice vector matrix given a description in terms of the
    lattice vector lengths and the angles in between.

    Args:
       a: First cell vector length.
       b: Second cell vector length.
       c: Third cell vector length.
       alpha: Angle between sides b and c in radians.
       beta: Angle between sides a and c in radians.
       gamma: Angle between sides b and a in radians.

    Returns:
       An array giving the lattice vector matrix in upper triangular form.
    """

    h = np.zeros((3, 3), float)
    h[0, 0] = a
    h[0, 1] = b * math.cos(gamma)
    h[0, 2] = c * math.cos(beta)
    h[1, 1] = b * math.sin(gamma)
    h[1, 2] = (b * c * math.cos(alpha) - h[0, 1] * h[0, 2]) / h[1, 1]
    h[2, 2] = math.sqrt(c**2 - h[0, 2] ** 2 - h[1, 2] ** 2)
    return h

# Example
# trajectory = AtomicStructures.from_file(file)
# structure  = trajectory[0]                               # --> ase.Atoms
# positions  = trajectory.positions                        # --> (N,3,3)
# dipole     = trajectory.call(lambda e: e.info["dipole"]) # --> (N,3)

class AtomicStructures(List[Atoms], pickleIO):
    """Class to handle atomic structures:
        - read from and write to big files (using `ase`)
        - serialization using `pickle`
        - automatic extraction of `info` and `array` from the list of structures
    """
    @classmethod
    @pickleIO.correct_extension_in
    def from_file(cls, **argv):
        """Load atomic structures from a file."""
        traj = read_trajectory(**argv)
        return cls(traj)
        
    @pickleIO.correct_extension_out
    def to_file(self: T, file: str, format: Union[str, None] = None):
        """Write atomic structures to a file."""
        write(images=self, filename=file, format=format)
    
    def to_list(self: T) -> List[Atoms]:
        return list(self)
    
    def call(self: T, func) -> np.ndarray:
        t = easyvectorize(Atoms)(self)
        return t.call(func)
    
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
    
#------------------------------------#

def info(t:AtomicStructures,name:str)->np.ndarray:
    # t = easyvectorize(Atoms)(t)
    return t.call(lambda e:e.info[name])
    
def array(t:AtomicStructures,name:str)->np.ndarray:
    # t = easyvectorize(Atoms)(t)
    return t.call(lambda e:e.arrays[name])

#------------------------------------#
# function to efficiently read atomic structure from a huge file

def read_trajectory(file:str,
               format:str=None,
               index:str=":",
               pbc:bool=True,
               same_cell:bool=False,
               remove_replicas:bool=False)->List[Atoms]:

    format = format.lower() if format is not None else None
    f = "extxyz" if format in ["i-pi","ipi"] else format
    remove_replicas = False if format not in ["i-pi","ipi"] else remove_replicas

    # with open(file,"r") as ffile:

    atoms = read(file,index=index,format=f)

    index = integer_to_slice_string(index)

    # comment = reconstruct_comment(atoms[0].info)
    if not isinstance(atoms,list):
        atoms = [atoms]
    for n in range(len(atoms)):
        # atoms[n].info = dict()
        atoms[n].calc = None # atoms[n].set_calculator(None)

    with open(file,"r") as ffile:

        pbc = pbc if pbc is not None else True
        if format in ["i-pi","ipi"]:
            for n in range(len(atoms)):
                atoms[n].info = dict()

            # try : 
            if same_cell:
                comment = read_comments_xyz(ffile,slice(0,1,None))[0]
                # ffile.readline() # first line
                # comment = ffile.readline()
                # ffile.seek(0)
                comments = FakeList(comment,len(atoms))
            else:
                comments = read_comments_xyz(ffile,index) #,Nread=len(atoms),snapshot_slice=index)
                if len(comments) != len(atoms):
                    raise ValueError("coding error: found comments different from atomic structures: {:d} comments != {:d} atoms (using index {})."\
                                     .format(len(comments),len(atoms),index))

            if pbc:
                strings = [ abcABC.search(comment) for comment in comments ]
                cells = np.zeros((len(strings),3,3))
                for n,cell in enumerate(strings):
                    a, b, c = [float(x) for x in cell.group(1).split()[:3]]
                    alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
                    cells[n] = abc2h(a, b, c, alpha, beta, gamma)

            if remove_replicas:
                if same_cell:
                    comments = read_comments_xyz(ffile,index)
                strings = [ step.search(comment).group(1) for comment in comments ]
                steps = np.asarray([int(i) for i in strings],dtype=int)
                test, indices = np.unique(steps, return_index=True)
                np.savetxt("steps-without-replicas.positions.txt",test,fmt="%d")
                # if steps.shape != test.shape:
                #     print("\t{:s}: there could be replicas. Specify '-rr/--remove_replicas true' to remove them.".format(warning))
                # if len(indices) != len(steps):
                #     pass
                atoms = [atoms[index] for index in indices]
                # for n in range(len(atoms)):
                #     atoms[n].info["step"] = indices[n]

            # matches = re.findall(abcABCunits,comments[0])
            # if len(matches) != 2 :
            #     raise ValueError("wrong number of matches")
            # else :
            #     units = {
            #         "positions" : matches[0],
            #         "cell" : matches[1]
            #     }
            
            for a in atoms:
                a.set_cell(cells[n].T if pbc else None)
                a.set_pbc(pbc)

        # if pbc:
        for a in atoms:
            # atoms[n].set_cell(cells[n].T)
            a.set_pbc(pbc)
            if not pbc:
                a.set_cell(None)


            # except:
            #     pass
    return atoms
    # return easyvectorize(Atoms)(atoms)

#------------------------------------#
def integer_to_slice_string(index):
    if isinstance(index, int):
        return string2index(f"{index}:{index+1}")
    elif index is None:
        return slice(None,None,None)
    elif isinstance(index, str):
        try:
            return string2index(index)
        except:
            raise ValueError("error creating slice from string {:s}".format(index))
    elif isinstance(index, slice):
        return index
    else:
        raise ValueError("`index` can be int, str, or slice, not {}".format(index))
    
#------------------------------------#
class FakeList:
    def __init__(self, value, length):
        self.value = value
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return [self.value] * ((stop - start + step - 1) // step)
        elif 0 <= index < self.length:
            return self.value
        else:
            raise IndexError("FakeList index out of range")