import functools
import glob
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Match, TypeVar, Union
import numpy as np
from ase import Atoms, io
from ase.cell import Cell
from ase.io.formats import filetype

from eslib.classes.file_formats.hdf5 import read_hdf5, write_hdf5
from eslib.classes.file_formats.pdb import read_pdb
from eslib.classes.io import pickleIO
from eslib.functions import extract_number_from_filename
from eslib.tools import convert
from eslib.io_tools import read_comments_ipi, integer_to_slice_string

T = TypeVar('T', bound='aseio')
M = TypeVar('M', bound=Callable[..., Any])

PARALLEL = False
CAST2LIST = True
ADD_FILE = False

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

                from eslib.classes.atomic_structures import AtomicStructures

                # Process each matched file
                N = len(matched_files)
                # print()
                for n, file in enumerate(matched_files):
                    # print(f"\t{n+1}/{N} : {file}                        ", end="\r",flush=True)
                    kwargs['file'] = file  # Update 'file' in kwargs
                    atoms:AtomicStructures = method(cls,*args, **kwargs)  # Call the method with updated 'file'
                    if ADD_FILE:
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
        
        if 'file' in argv and isinstance(argv['file'], str):
            file = argv['file']
            _,format = os.path.splitext(file)
            format = format[1:]
        if format.lower() in ["h5","hdf5"]:
            index = argv['index'] if 'index' in argv else slice(None,None,None)
            index = integer_to_slice_string(index)
            traj = read_hdf5(filename=file,index=index)
        # elif format.lower() in ["nc","netcdftrajectory"]:
        #     warn("NetCDF trajectory format is deprecated. Use 'hdf5/h5' format instead.",DeprecationWarning)
        #     index = argv['index'] if 'index' in argv else slice(None,None,None)
        #     index = integer_to_slice_string(index)
        #     traj = read_netcdftrajectory(filename=argv['file'],index=index)    
        #     if not isinstance(traj,list):
        #         traj = [traj]             
        # elif format.lower() in ["netcdf"]: 
        #     warn("NetCDF trajectory format is deprecated. Use 'hdf5/h5' format instead.",DeprecationWarning)
        #     traj = read_netcdf(file)
        elif format.lower() in ["pdb"]:
            traj = [read_pdb(file)]
        else:
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
        if format is None:
            try:
                format = filetype(file, read=False)
            except:
                _,format = os.path.splitext(file)
        if format in ["ipi","i-pi"]:
            if os.path.exists(file):
                os.remove(file)
            for atoms in self:
                params = atoms.get_cell().cellpar()
                float_format = '%15.10e'
                fmt_header = "# CELL(abcABC): {:s}  {:s}  {:s}  {:s}  {:s}  {:s}  %s".format(*([float_format]*6))
                string = " positions{angstrom} cell{angstrom}"
                comment = fmt_header%(*params,string)
                io.write(file, atoms, format="xyz", append=True, comment=comment)
        # elif format.lower() in ["nc","netcdftrajectory"]:
        #     write_netcdftrajectory(filename=file,images=self.to_list())
        # elif format.lower() in ["netcdf"]:
        #     write_netcdf(file,self.to_list())
        elif format.lower() in ["h5","hdf5"]:
            write_hdf5(file,self.to_list())
        else:
            io.write(images=self, filename=file, format=format)

    #------------------#
    def to_pickle(self:T, file:str)->None:
        """Save the object to a `*.pickle` file after casting it to `List[Atoms]`.

        This is supposed to help with backward compatibilty and data transferability.
        
        In fact the object that is saved to file is only the 'core data structure'.
        """
        if CAST2LIST:
            obj = self.to_list()
        else:
            obj = self
        pickleIO.to_pickle(obj,file)
    
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

    if format in [None,"xyz","extxyz"]:
        # `extxyz.read`` from https://github.com/libAtoms/extxyz
        # this function should be faster than `ASE.io.read``
        try:
            from extxyz import read
            atoms = read(file,index=index)
            if len(atoms) == 0 :
                raise ValueError("some error occurred")
        except:
            atoms = io.read(file,index=index,format=f)
    else:
        # ASE.io.read
        atoms = io.read(file,index=index,format=f)
        
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
            comment = read_comments_ipi(ffile,slice(0,1,None))[0]
            units["positions"] = str(comment).split("positions{")[1].split("}")[0]
            units["cell"] = str(comment).split("cell{")[1].split("}")[0]
            factor = {
                "positions" : convert(1,"length",_from=units["positions"],_to="angstrom"),
                     "cell" : convert(1,"length",_from=units["cell"],     _to="angstrom"),
            }

            if same_cell:
                # comment = read_comments_ipi(ffile,slice(0,1,None))[0]
                comments = FakeList(comment,len(atoms))
            else:
                comments = read_comments_ipi(ffile,index)
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
            else:
                cells:List[Union[Cell, None]] = FakeList(None,len(atoms))

            if remove_replicas:
                if not read_all_comments:
                    comments = read_comments_ipi(ffile,index)
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

            for atom, cell in zip(atoms,cells):
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
class FakeList:
    """
    A fake list implementation.
    """
    def __init__(self, value: Any, length: int):
        self.value = value
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: Union[int, slice]) -> Union[Any, List[Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return [self.value] * ((stop - start + step - 1) // step)
        elif 0 <= index < self.length:
            return self.value
        else:
            raise IndexError("FakeList index out of range")
