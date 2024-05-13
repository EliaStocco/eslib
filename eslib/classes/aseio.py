from ase.io import read, write, string2index
from ase import Atoms
from io import TextIOWrapper
import re
import math
import numpy as np
from eslib.classes.io import pickleIO
from typing import List, Union, TypeVar

T = TypeVar('T', bound='aseio')

#---------------------------------------#
class aseio(List[Atoms], pickleIO):
    """Class to handle atomic structures:
        - read from and write to big files (using `ase`)
        - serialization using `pickle`
    """
    @classmethod
    @pickleIO.correct_extension_in
    def from_file(cls, **argv):
        """
        Load atomic structures from file.

        Attention: it's recommended to use keyword-only arguments.
        """
        traj = read_trajectory(**argv)
        return cls(traj)
        
    @pickleIO.correct_extension_out
    def to_file(self: T, file: str, format: Union[str, None] = None):
        """
        Write atomic structures to file.
        
        Attention: it's recommended to use keyword-only arguments.
        """
        write(images=self, filename=file, format=format)
    
    def to_list(self: T) -> List[Atoms]:
        return list(self)
    
#------------------------------------#
# function to efficiently read atomic structure from a huge file
def read_trajectory(file:str,
               format:str=None,
               index:str=":",
               pbc:bool=True,
               same_cell:bool=False,
               remove_replicas:bool=False)->List[Atoms]:
    """
    Read atomic structures from a trajectory file.

    Args:
        file (str): Path to the trajectory file.
        format (str, optional): File format. Defaults to None.
        index (str, optional): Selection string. Defaults to ":".
        pbc (bool, optional): Periodic boundary conditions. Defaults to True.
        same_cell (bool, optional): Use the same cell for all frames. Defaults to False.
        remove_replicas (bool, optional): Remove replicas. Defaults to False.

    Returns:
        List[Atoms]: List of atomic structures.
    """

    format = format.lower() if format is not None else None
    f = "extxyz" if format in ["i-pi","ipi"] else format
    remove_replicas = False if format not in ["i-pi","ipi"] else remove_replicas

    atoms = read(file,index=index,format=f)
    index = integer_to_slice_string(index)

    if not isinstance(atoms,list):
        atoms = [atoms]
    for n in range(len(atoms)):
        atoms[n].calc = None 
    with open(file,"r") as ffile:

        pbc = pbc if pbc is not None else True
        if format in ["i-pi","ipi"]:
            for n in range(len(atoms)):
                atoms[n].info = dict()

            # try : 
            if same_cell:
                comment = read_comments_xyz(ffile,slice(0,1,None))[0]
                comments = FakeList(comment,len(atoms))
            else:
                comments = read_comments_xyz(ffile,index)
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
                # np.savetxt("steps-without-replicas.positions.txt",test,fmt="%d")
                atoms = [atoms[index] for index in indices]


            for a in atoms:
                a.set_cell(cells[n].T if pbc else None)
                a.set_pbc(pbc)

        for a in atoms:
            a.set_pbc(pbc)
            if not pbc:
                a.set_cell(None)

    return atoms

#------------------------------------#
deg2rad     = np.pi / 180.0
abcABC      = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
abcABCunits = re.compile(r'\{([^}]+)\}')
step        = re.compile(r"Step:\s+(\d+)")

#------------------------------------#
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

#------------------------------------#
def is_convertible_to_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#------------------------------------#
def integer_to_slice_string(index):
    """
    Convert integer index to slice string.

    Args:
        index: Index to convert.

    Returns:
        slice: Converted slice.
    """
    
    if is_convertible_to_integer(index):
        index=int(index)

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

#---------------------------------------#
def get_offset(file:TextIOWrapper,
               Nmax:int=1000000,
               line_offset_old:list=None,
               index:slice=None):
    """
    Get line offsets in a file.

    Args:
        file (TextIOWrapper): File object.
        Nmax (int, optional): Maximum number of offsets. Defaults to 1000000.
        line_offset_old (list, optional): Old line offsets. Defaults to None.
        index (slice, optional): Slice index. Defaults to None.

    Returns:
        list: List of line offsets.
    """
    # Read in the file once and build a list of line offsets
    n = 0 
    line_offset = [None]*Nmax
    if line_offset_old is not None:
        line_offset[:len(line_offset_old)] = line_offset_old
        n = len(line_offset_old)
        del line_offset_old
    offset = 0
    restart = False
    if n == 0 : file.seek(0) # start from the beginning of the file
    for line in file: # cycle over all lines
        if n >= Nmax: # create a bigger list
            restart = True
            break
        if line.replace(" ","").startswith("#"): # check if the line contains a comment
            line_offset[n] = offset 
            n += 1
        if index is not None and index.stop is not None and n >= index.stop: # stop
            break
        offset += len(line)        
    if restart: return get_offset(file, Nmax=Nmax * 2, line_offset_old=line_offset, index=index)
    file.seek(0)
    return line_offset[:n]

#---------------------------------------#
def read_comments_xyz(file:TextIOWrapper,
                      index:slice=None):
    """
    Read comments from an XYZ file.

    Args:
        file (TextIOWrapper): File object.
        index (slice, optional): Slice index. Defaults to None.

    Returns:
        list: List of comments.
    """
    offset = get_offset(file=file,index=index)
    if index is not None:
        offset = offset[index]
    comments = [None]*len(offset)
    for n,l in enumerate(offset):
        file.seek(l)
        comments[n] = file.readline()
    return comments

#------------------------------------#
class FakeList:
    """
    A fake list implementation.
    """
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