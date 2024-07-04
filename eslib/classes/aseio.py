from ase import Atoms
from io import TextIOWrapper
import re
import os
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Union, TypeVar, Match, Callable, Any, Dict
import functools
import glob
from ase.io import read, write, string2index
# from classes.trajectory import AtomicStructures
from eslib.classes.io import pickleIO
from eslib.tools import convert
from eslib.functions import extract_number_from_filename
# from eslib.formatting import float_format

T = TypeVar('T', bound='aseio')
M = TypeVar('M', bound=Callable[..., Any])

PARALLEL = False

#------------------#
def set_parallel(value: bool):
    """
    Set the global PARALLEL flag.

    Parameters:
    value (bool): The new value for the PARALLEL flag.
    """
    global PARALLEL
    PARALLEL = value

#------------------#
def calc_none_static(method: M) -> M:
    """
    Decorator to set `calc` = None for all `ase.Atoms` objects in `self`
    before executing the method (usually before saving the object to file).

    Parameters:
    method (Callable): The method to be decorated.

    Returns:
    Callable: The wrapped method with calc set to None.

    Attention:
    It is very important in post-processing scripts to have `calc` = None.
    This function assures that this will be the case.
    Weird behaviors (hard to detect and debug) can occur if  `calc` != None, especially when IO streaming.
    """
    @functools.wraps(method)
    def wrapper(self: List[Atoms], *args, **kwargs) -> Any:
        for a in self:
            a.calc = None
        return method(self, *args, **kwargs)
    return wrapper

#------------------#
def calc_none_class(method: M) -> M :
    """
    Decorator to set `calc` = None for all `ase.Atoms` objects in the output
    returned by the class method.

    Parameters:
    method (Callable): The method to be decorated.

    Returns:
    List[Atoms]: The list of `ase.Atoms` objects with `calc` set to None.

    Attention:
    It is very important in post-processing scripts to have `calc` = None.
    This function assures that this will be the case.
    Weird behaviors (hard to detect and debug) can occur if  `calc` != None, especially when IO streaming.
    """
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        out: List[Atoms] = method(*args, **kwargs)
        for a in out:
            a.calc = None
        return out
    return wrapper

#------------------#
def correct_pbc(atoms:List[Atoms]):
    for a in atoms:
        if a.get_cell() is None or a.get_cell().volume == 0.0 :
            a.set_pbc(False)
        else:
            a.set_pbc(True)

#------------------#
def correct_pbc_static(method: M) -> M:
    @functools.wraps(method)
    def wrapper(self: List[Atoms], *args, **kwargs) -> Any:
        correct_pbc(self)
        return method(self, *args, **kwargs)
    return wrapper

#------------------#
def correct_pbc_class(method: M) -> M :
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        out: List[Atoms] = method(*args, **kwargs)
        correct_pbc(out)
        return out
    return wrapper

#------------------#
def file_pattern(method: M) -> M:
    """
    A decorator that modifies the behavior of a method to support file pattern matching.

    If the keyword argument 'file' is provided and contains a file pattern (e.g., '*.txt'), 
    the decorator will use the `glob` module to find all matching files. It will then 
    call the decorated method for each matched file and collect the results into a single list.

    If no files match the pattern, a ValueError is raised.

    Args:
        method (M): The method to be decorated.

    Returns:
        M: The wrapped method with added file pattern matching capability.
    """
    
    @functools.wraps(method)
    def wrapper(cls,*args, **kwargs) -> Any:
        # Check if 'file' keyword argument is present
        if 'file' in kwargs:
            # Find all files matching the pattern
            matched_files = glob.glob(kwargs['file'])
            
            # Raise an error if no files were found
            if not matched_files:
                raise ValueError("No files found")
            
            if len(matched_files) > 1:            
                try:
                    matched_files = [ matched_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in matched_files ])) ]
                except:
                    pass
                
                # Initialize a list to hold results from each file
                structures = [None] * len(matched_files)

                from classes.trajectory import AtomicStructures            
                # Process each matched file
                for n, file in enumerate(matched_files):
                    kwargs['file'] = file  # Update 'file' in kwargs
                    atoms:AtomicStructures = method(cls,*args, **kwargs)  # Call the method with updated 'file'
                    atoms.set("file",np.array([file] * len(atoms)),"info")
                    structures[n] = atoms.copy()
                
                # Flatten the list of results
                return cls([item for sublist in structures for item in sublist])
        
        # Call the method normally if 'file' is not in kwargs
        return method(cls,*args, **kwargs)

    return wrapper

#------------------#
class aseio(List[Atoms], pickleIO):
    """Class to handle atomic structures:
        - read from and write to big files (using `ase`)
        - serialization using `pickle`
    """
    
    #------------------#
    # Attention!
    # The order of the following decorators matters.
    # Do not change it.
    @classmethod
    @file_pattern
    @correct_pbc_class
    @calc_none_class # try to comment it if you encounter ay problem
    @pickleIO.correct_extension_in
    def from_file(cls, **argv):
        """
        Load atomic structures from file.

        Attention: it's recommended to use keyword-only arguments.
        """
        traj = read_trajectory(**argv)
        return cls(traj)
    
    #------------------#
    # Attention!
    # The order of the following decorators matters.
    # Do not change it.
    @correct_pbc_static
    @calc_none_static # try to comment it if you encounter ay problem
    @pickleIO.correct_extension_out
    def to_file(self: T, file: str, format: Union[str, None] = None):
        """
        Write atomic structures to file.
        
        Attention: it's recommended to use keyword-only arguments.
        """
        if format in ["ipi","i-pi"]:
            if os.path.exists(file):
                os.remove(file)
            for atoms in self:
                params = atoms.get_cell().cellpar()
                float_format = '%15.10e'
                fmt_header = "# CELL(abcABC): {:s}  {:s}  {:s}  {:s}  {:s}  {:s}  %s".format(*([float_format]*6))
                string = " positions{angstrom} cell{angstrom}"
                comment = fmt_header%(*params,string)
                write(file, atoms, format="xyz", append=True, comment=comment)
        else:
            write(images=self, filename=file, format=format)
    
    def to_list(self: T) -> List[Atoms]:
        return list(self)
    
#------------------#
deg2rad     = np.pi / 180.0
abcABC      = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
abcABCunits = re.compile(r'\{([^}]+)\}')
Step        = re.compile(r"Step:\s+(\d+)")
    
#------------------#
# function to efficiently read atomic structure from a huge file
def read_trajectory(file:str,
               format:str=None,
               index:str=":",
               pbc:bool=None,
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

    units = {
        "positions" : "atomic_unit",
        "cell" : "atomic_unit"
    }
    factor = {
        "positions" : np.nan,
        "cell" : np.nan
    }

    read_all_comments = False

    if not isinstance(atoms,list):
        atoms = [atoms]

    ########################
    # Attention:
    # the following line is MANDATORY
    # if tou do not set `calc`=None in post-processing script you could get 
    # really weird (and wrong) behavior in the IO stream
    for atom in atoms:
        if atom.calc is not None:
            results:Dict[str,Union[float,np.ndarray]] = atom.calc.results
            for key,value in results.items():
                if key in ['energy','free_energy','dipole','stress']:
                    atom.info[key] = value
                elif key in ['forces']:
                    atom.arrays[key] = value
                else: 
                    atom.info[key] = value
        atom.calc = None 
        if format in ["i-pi","ipi"]:
            atom.info = dict()
    ########################

    if format in ["i-pi","ipi"]:

        # for atom in atoms:
        #     atom.info = dict()

        pbc = pbc if pbc is not None else True

        with open(file,"r") as ffile:

            # units
            comment = read_comments_xyz(ffile,slice(0,1,None))[0]
            units["positions"] = str(comment).split("positions{")[1].split("}")[0]
            units["cell"] = str(comment).split("cell{")[1].split("}")[0]
            factor = {
                "positions" : convert(1,"length",_from=units["positions"],_to="angstrom"),
                     "cell" : convert(1,"length",_from=units["cell"],     _to="angstrom"),
            }

            if same_cell:
                # comment = read_comments_xyz(ffile,slice(0,1,None))[0]
                comments = FakeList(comment,len(atoms))
            else:
                comments = read_comments_xyz(ffile,index)
                read_all_comments = True
                if len(comments) != len(atoms):
                    raise ValueError("coding error: found comments different from atomic structures: {:d} comments != {:d} atoms (using index {})."\
                                        .format(len(comments),len(atoms),index))

            if pbc:
                if PARALLEL:
                    with ProcessPoolExecutor() as executor:
                        cells = executor.map(parallel_abcABC, comments)
                    cells = np.array(cells)
                else:
                    strings:List[Match[str]] = [ abcABC.search(comment) for comment in comments ]
                    cells = np.zeros((len(strings),3,3))
                    for n,string in enumerate(strings):
                        a, b, c = [float(x) for x in string.group(1).split()[:3]]
                        alpha, beta, gamma = [float(x) * deg2rad for x in string.group(1).split()[3:6]]
                        cells[n] = abc2h(a, b, c, alpha, beta, gamma)

            if remove_replicas:
                if not read_all_comments:
                    comments = read_comments_xyz(ffile,index)
                if PARALLEL:
                    with ProcessPoolExecutor() as executor:
                        steps = executor.map(parallel_steps, comments)
                        steps = np.array(steps,dtype=int)
                else:
                    strings:List[Match[str]] = [ Step.search(comment).group(1) for comment in comments ]
                    steps = np.asarray([int(i) for i in strings],dtype=int)

                unique_steps, indices = np.unique(steps, return_index=True)
                assert np.allclose(unique_steps,steps[indices])
                assert np.allclose(np.arange(len(unique_steps)),unique_steps)
                # np.savetxt("steps-without-replicas.positions.txt",test,fmt="%d")
                atoms:List[Atoms] = [atoms[index] for index in indices]
                for atom,step in zip(atoms,unique_steps):
                    atom.info["step"] = step

            for atom,cell in zip(atoms,cells):
                atom.set_cell(cell.T * factor["cell"] if pbc else None)
                atom.set_pbc(pbc)
                atom.positions *= factor["positions"]

    else:
        if pbc is not None:
            for atom in atoms:
                atom.set_pbc(pbc)
                if not pbc:
                    atom.set_cell(None)

    return atoms

#------------------#
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

#------------------#
def parallel_abcABC(comment: str):
    """
    Extract lattice parameters and compute the lattice vector matrix from a comment string.

    Args:
        comment (str): The comment string containing lattice parameters.

    Returns:
        np.ndarray: The lattice vector matrix.
    """
    cell = abcABC.search(comment)
    if cell:
        a, b, c = [float(x) for x in cell.group(1).split()[:3]]
        alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
        return abc2h(a, b, c, alpha, beta, gamma)
    else:
        raise ValueError("Invalid comment format")

#------------------#
def parallel_steps(comment: str)->int:
    """
    Extract step from a comment string.

    Args:
        comment (str): The comment string containing lattice parameters.

    Returns:
        step: the MD step
    """
    string:Match[str] = Step.search(comment).group(1)
    if string:
        return int(string)
    else:
        raise ValueError("Invalid comment format")

#------------------#
def is_convertible_to_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#------------------#
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

#------------------#
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

#------------------#
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

#------------------#
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