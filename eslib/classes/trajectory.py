from ase.io import read, string2index
from ase import Atoms
from .vectorize import easyvectorize
from eslib.functions import read_comments_xyz
import re
import ipi.utils.mathtools as mt
import numpy as np
from eslib.formatting import warning
from eslib.classes.io import pickleIO
from typing import List

deg2rad     = np.pi / 180.0
abcABC      = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
abcABCunits = re.compile(r'\{([^}]+)\}')
step        = re.compile(r"Step:\s+(\d+)")

# Example
# trajectory = AtomicStructures.from_file(file)
# structure  = trajectory[0]                               # --> ase.Atoms
# positions  = trajectory.positions                        # --> (N,3,3)
# dipole     = trajectory.call(lambda e: e.info["dipole"]) # --> (N,3)


# def info(t,name):
#     return t.call(lambda e:e.info[name])
# def array(t,name):
#     return t.call(lambda e:e.arrays[name])

def integer_to_slice_string(index):
    if isinstance(index, int):
        return string2index(f"{index}:{index+1}")
    elif index is None:
        return slice(None,None,None)
    elif isinstance(index, str):
        try:
            index = string2index(index)
        except:
            raise ValueError("error creating slice from string {:s}".format(index))
    elif isinstance(index, slice):
        return index
    else:
        raise ValueError("`index` can be int, str, or slice, not {}".format(index))
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
                    raise ValueError("coding error: found comments different from atomic structures.")

            if pbc:
                strings = [ abcABC.search(comment) for comment in comments ]
                cells = np.zeros((len(strings),3,3))
                for n,cell in enumerate(strings):
                    a, b, c = [float(x) for x in cell.group(1).split()[:3]]
                    alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
                    cells[n] = mt.abc2h(a, b, c, alpha, beta, gamma)

            if remove_replicas:
                if same_cell:
                    comments = read_comments_xyz(ffile,index)
                strings = [ step.search(comment).group(1) for comment in comments ]
                steps = np.asarray([int(i) for i in strings],dtype=int)
                test, indices = np.unique(steps, return_index=True)
                if steps.shape != test.shape:
                    print("\t{:s}: there could be replicas. Specify '-rr/--remove_replicas true' to remove them.".format(warning))
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
            if pbc:
                for a in atoms:
                    a.set_cell(cells[n].T)
                    # a.set_pbc(True)

        if pbc:
            for a in atoms:
                # atoms[n].set_cell(cells[n].T)
                a.set_pbc(True)


            # except:
            #     pass
    return atoms
    # return easyvectorize(Atoms)(atoms)


class AtomicStructures(list,pickleIO):
    """Class to handle atomic structures:
        - read from and write to big files (using `ase`)
        - serialization using `pickle`
        - automatic extraction of `info` and `array` from the list of structures
    """
    @classmethod
    def from_file(cls,**argv)->List[Atoms]:
        if 'file' in argv and isinstance(argv['file'], str) and argv['file'].endswith('.pickle'):
            return AtomicStructures.from_pickle(argv['file'])
        else:
            traj = read_trajectory(**argv)
            return cls(traj)
    
    def to_list(self):
        return list(self)
    
    def call(self,func):
        t = easyvectorize(Atoms)(self)
        return t.call(func)

def info(t:AtomicStructures,name:str):
    # t = easyvectorize(Atoms)(t)
    return t.call(lambda e:e.info[name])
    
def array(t:AtomicStructures,name:str):
    # t = easyvectorize(Atoms)(t)
    return t.call(lambda e:e.arrays[name])