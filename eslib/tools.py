import glob
import re
from copy import copy
from glob import glob
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from ase import Atoms
from ase.cell import Cell
from ase.geometry import distance
try:
    from phonopy.structure.atoms import PhonopyAtoms
except:
    pass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from eslib.ipi_units import UnitMap, unit_to_internal, unit_to_user


def take(arr, axis, indices,keepdims=False):
    # Take the values along the specified axis
    result = np.take(arr, indices=indices, axis=axis)
    
    # If indices is a scalar, add back the singleton dimension along the axis
    if np.isscalar(indices) and keepdims:
        result = np.expand_dims(result, axis=axis)
    
    return result

def get_files(pattern: str):
    # Get the list of files using glob
    files = glob.glob(pattern)

    # Function to extract the first number from the filename
    def extract_number(filename):
        # This regex will extract the first sequence of digits found in the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))  # Return the number as an integer
        return float('inf')  # If no number is found, return a large number

    # Sort the files based on the extracted number
    try:
        return sorted(files, key=extract_number)
    except:
        return files


#---------------------------------------#
def find_transformation(A: Atoms, B: Atoms):
    """
    Compute the transformation matrix between the cells/lattice vectors of two atomic structures.

    Parameters:
        A (ase.Atoms): The first atomic structure (primitive cell).
        B (ase.Atoms): The second atomic structure (supercell).

    Returns:
        numpy.ndarray: The transformation matrix from A to B.
    """
    # Compute the transformation matrix
    M = B.get_cell().T @ np.linalg.inv(A.get_cell().T)

    if not np.allclose(B.get_cell().T,M @ A.get_cell().T):
        raise ValueError("error in the code implementation")

    return M

#---------------------------------------#
def segment(A:np.ndarray, B:np.ndarray, N:int, start:int=0, end:int=1):
    """This function generates a segment
    given the initial (A) and final (B) points
    and put N points in the middle.

    A and B can be any kind of np.ndarray
    """
    assert A.shape == B.shape

    sequence = np.zeros((N + 2, *A.shape))
    T = np.linspace(start, end, N + 2)
    # N = 0 -> t=0,1
    # N = 1 -> t=0,0.5,1
    for n, t in enumerate(T):
        # t = float(n)/(N+1)
        sequence[n] = A * (1 - t) + t * B
    return sequence

#---------------------------------------#
def get_sorted_atoms_indices(reference:Atoms,structure:Atoms):
    """Calculate pairwise distances and obtain optimal sorting indices."""
    # Calculate the pairwise distances between atoms in the two structures
    distances = cdist(reference.get_positions(), structure.get_positions())
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distances)
    return col_ind

#---------------------------------------#
def sort_atoms(reference:Atoms,structure:Atoms):
    """Sort atoms in the second structure by minimizing the distances w.r.t. the atoms in the first structure."""
    indices = get_sorted_atoms_indices(reference, structure)
    sorted = structure[indices]
    return sorted, indices

#---------------------------------------#
def find_transformation(A:Atoms,B:Atoms):
    """Compute the transformation matrix between the lattice vectors of two atomic structures."""
    M = np.asarray(B.cell).T @ np.linalg.inv(np.asarray(A.cell).T)
    size = M.round(0).diagonal().astype(int)
    return size, M

#---------------------------------------#
def convert(what:Union[np.ndarray,float], family:str=None, _from:str="atomic_unit", _to:str="atomic_unit")->Union[np.ndarray,float]:
    """Convert a quantity from one unit of a specific family to another.
    Example: 
    arr = convert([1,3,4],'length','angstrom','atomic_unit')
    arr = convert([1,3,4],'energy','atomic_unit','millielectronvolt')"""
    # from ipi.utils.units import unit_to_internal, unit_to_user
    if family is not None:
        factor = unit_to_internal(family, _from, 1)
        factor *= unit_to_user(family, _to, 1)
        return what * factor
    else :
        return what

#---------------------------------------#
def add_conversion(family:str,unit:str,value:float)->None:
    """value must be the conversion factor from `unit` to 'atomic_unit'"""
    UnitMap[family.lower()][unit.lower()] = value

value = convert(1,"charge","coulomb","atomic_unit")/( convert(1,"length","meter","atomic_unit")**2)
add_conversion("polarization","C/m^2",value)

# value = convert(1,"electric-field","V/ang","atomic_unit")*convert(1,"length","centimeter","angstrom")# *convert(1,"length","centimeter","atomic_unit"))
# add_conversion("electric-field","V/cm",value)
value = convert(1,"electric-field","V/ang","atomic_unit")
add_conversion("electric-field","MV/cm",1e-2*value)

#---------------------------------------#
# Decorator to convert ase.Cell to np.array and transpose
def ase_cell_to_np_transpose(func):
    """Decorator to convert an ASE cell to NumPy array and transpose for use in the decorated function."""
    def wrapper(cell:Union[np.ndarray,Cell], *args, **kwargs):
        if isinstance(cell, Cell):
            cell = np.asarray(cell).T
        if cell.shape != (3,3):
            raise ValueError("cell with wrong shape:",cell.shape)
        return func(cell, *args, **kwargs)
    return wrapper

#---------------------------------------#
def return_transformed_components(func):
    """Decorator to automatically compute the matrix multiplication if a vector is provided."""
    def wrapper(cell:Union[np.ndarray,Cell],v:np.ndarray=None, *args, **kwargs)->np.ndarray:
        matrix = func(cell=cell,*args, **kwargs)
        if not isinstance(matrix,np.ndarray):
            raise TypeError("'matrix' should be 'np.ndarray'")
        if v is None:
            return matrix
        else:
            shape = v.shape
            v = np.asarray(v).reshape((-1,3))
            out = (matrix @ v.T).T
            return out.reshape(shape)
    return wrapper

#---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def lattice2cart(cell:Union[np.ndarray,Cell],v:np.ndarray=None)->np.ndarray:
    """ Lattice to Cartesian coordinates rotation matrix."""
    from copy import copy

    # normalize the lattice parameters
    length = np.linalg.norm(cell,axis=0)
    matrix = copy(cell)
    # normalize the columns
    for i in range(3):
        matrix[:,i] /= length[i]
    return matrix

#---------------------------------------# 
@return_transformed_components
@ase_cell_to_np_transpose
def cart2lattice(cell:Union[np.ndarray,Cell],v:np.ndarray=None)->np.ndarray:
    """ Cartesian to lattice coordinates rotation matrix."""
    matrix = lattice2cart(cell)
    matrix = np.linalg.inv(matrix)
    return matrix

#---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def frac2cart(cell:Union[np.ndarray,Cell],v:np.ndarray=None)->np.ndarray:
    """Return a 3x3 rotation matrix from fractional to cartesian coordinates.\n
    The lattice vectors are not normalized, differently w.r.t. `lattice2cart`.\n
    The function is decorated such that you can provide vector `v` as `numpy.ndarray` to automatically rotate them.
    
    Input:
        `cell`: lattice parameters 
            a `numpy.ndarray` with the i^th lattice vector stored in the i^th columns (it's the opposite of ASE, QE, FHI-aims)\n
            or an `ase.cell.Cell` object (the code will automatically convert it to `numpy.ndarray` and take the transpose).\n
            The format of `cell` should be the following: 
                | a_1x a_2x a_3x |\n
                | a_1y a_2y a_3y |\n
                | a_1z a_2z a_3z |\n
        `v`: vectors (optional)
            `numpy.ndarray` with the vectors to rotate.

    Output:
        rotation matrix: `numpy.ndarray` of shape 3x3,\n
        or rotated vectors if `v` is provided: `numpy.ndarray` with the same shape of `v`
    """
    return cell

#---------------------------------------# 
@return_transformed_components
@ase_cell_to_np_transpose
def cart2frac(cell:Union[np.ndarray,Cell],v:np.ndarray=None)->np.ndarray:
    """ Cartesian to lattice coordinates rotation matrix."""
    matrix = frac2cart(cell)
    matrix = np.linalg.inv(matrix)
    return matrix

@ase_cell_to_np_transpose
def max_pbc_distance(cell:Union[np.ndarray,Cell])->float:
    vec = frac2cart(cell,np.asarray([0.5,0.5,0.5]))
    return np.linalg.norm(vec)

#---------------------------------------#
def project_on_Miller(cell:np.ndarray, Miller:np.ndarray,array:np.ndarray)->np.ndarray:
    """
    Projects an array onto a Miller index direction in lattice coordinates.

    This function takes a crystal cell, a Miller index, and an array, 
    converts the array from cartesian to lattice coordinates, and then 
    projects this array onto the specified Miller index direction.
    The output has to be intended with lenght dimensions, same as the input array.

    Parameters:
    - cell (np.ndarray): The lattice parameters of the crystal cell.
    - Miller (np.ndarray): A 1D array representing the Miller index 
      direction to project onto. It will be normalized internally.
    - array (np.ndarray): The input array in cartesian coordinates.

    Returns:
    - np.ndarray: The array projected onto the Miller index direction 
      in lattice coordinates.
    """
    Miller= np.asarray(Miller,dtype=float)
    assert Miller.ndim == 1, "Miller must be a 1D array"
    Miller /= np.linalg.norm(Miller)
    array = cart2lattice(cell,array)
    array = np.dot(array,Miller)
    # return frac2cart(cell,array)
    return array

def relative_vectors(structure:Atoms,_all=False):
    columns = ["i","j","Si","Sj", # indices and symbols
               "uwdx","uwdy","uwdz","uwf1","uwf2","uwf3", # unwrapped vectors and unwrapped fractional vectors
               "wdx","wdy","wdz","wf1","wf2","wf3", # wrapped vectors and wrapped fractional vectors
               "uwd","uwf","wd","wf"]  # unwrapped and wrapped distances and fractional distances
    df = pd.DataFrame(columns=columns)
    
    N = structure.get_global_number_of_atoms()  # Example value of N
    a = np.arange(N)
    # Generate all pairs (i, j) where i ranges from 0 to N-1 and j ranges from i+1 to N-1
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    mask = i < j
    pairs = np.column_stack((i[mask], j[mask]))
    assert pairs.shape[0] == int(N*(N-1)/2)

    df["i"] = pairs[:,0]
    df["j"] = pairs[:,1]

    symbols = structure.get_chemical_symbols()
    df["Si"] = [ symbols[i] for i in df["i"] ]
    df["Sj"] = [ symbols[i] for i in df["j"] ]

    from icecream import ic

    pos = structure.get_positions()
    vec = np.asarray([ pos[i,:] - pos[j,:]  for i,j in zip(df["i"],df["j"]) ])
    df["uwdx"] = vec[:,0]
    df["uwdy"] = vec[:,1]
    df["uwdz"] = vec[:,2]
    df["uwd"]  = np.linalg.norm(vec,axis=1)

    fvec = cart2frac(structure.get_cell(),vec)
    df["uwfx"] = fvec[:,0]
    df["uwfy"] = fvec[:,1]
    df["uwfz"] = fvec[:,2]
    df["uwf"]  = np.linalg.norm(fvec,axis=1)

    wfvec = np.mod(fvec,1) - 0.5
    df["wfx"] = wfvec[:,0]
    df["wfy"] = wfvec[:,1]
    df["wfz"] = wfvec[:,2]
    df["wf"]  = np.linalg.norm(wfvec,axis=1)

    wvec = frac2cart(structure.get_cell(),wfvec)
    df["wdx"] = wvec[:,0]
    df["wdy"] = wvec[:,1]
    df["wdz"] = wvec[:,2]
    df["wd"]  = np.linalg.norm(wvec,axis=1)

    if _all:
        return df
    else:
        newdf = df[["i","j","Si","Sj","wdx","wdy","wdz","wd"]]
        new_column_names = {'wdx': 'x', 'wdy': 'y', 'wdz': 'z', 'wd':'d'}
        return newdf.rename(columns=new_column_names)



#---------------------------------------#
def string2function(input_string:str)->callable:
    """Converts a Python code string into a callable function."""
    import ast

    # Parse the input string as Python code
    parsed_code = ast.parse(input_string, mode='eval')
    # Create a function from the parsed code
    code_object = compile(parsed_code, filename='<string>', mode='eval')
    function = eval(code_object)
    return function

#---------------------------------------#
def distance(s1:Atoms, s2:Atoms, permute=True):
    """Get the distance between two structures s1 and s2.
    
    The distance is defined by the Frobenius norm of
    the spatial distance between all coordinates (see
    numpy.linalg.norm for the definition).

    permute: minimise the distance by 'permuting' same elements
    """

    s1 = s1.copy()
    s2 = s2.copy()
    for s in [s1, s2]:
        s.translate(-s.get_center_of_mass())
    s2pos = 1. * s2.get_positions()
    
    def align(struct:Atoms, xaxis='x', yaxis='y'):
        """Align moments of inertia with the coordinate system."""
        Is, Vs = struct.get_moments_of_inertia(True)
        IV = list(zip(Is, Vs))
        IV.sort(key=lambda x: x[0])
        struct.rotate(IV[0][1], xaxis, rotate_cell=True)
        
        Is, Vs = struct.get_moments_of_inertia(True)
        IV = list(zip(Is, Vs))
        IV.sort(key=lambda x: x[0])
        struct.rotate(IV[1][1], yaxis, rotate_cell=True)

    # align(s1)

    def dd(s1:Atoms, s2:Atoms, permute):
        if permute:
            s2 = s2.copy()
            dist = 0
            for a in s1:
                imin = None
                dmin = np.Inf
                for i, b in enumerate(s2):
                    if a.symbol == b.symbol:
                        d = np.sum((a.position - b.position)**2)
                        if d < dmin:
                            dmin = d
                            imin = i
                dist += dmin
                s2.pop(imin)
            return np.sqrt(dist)
        else:
            return np.linalg.norm(s1.get_positions() - s2.get_positions())

    dists = []
    # principles
    for x, y in zip(['x', '-x', 'x', '-x'], ['y', 'y', '-y', '-y']):
        s2.set_positions(s2pos)
        align(s2, x, y)
        dists.append(dd(s1, s2, permute))
   
    return min(dists), s1, s2

def check_cell_format(cell:Union[np.ndarray,Cell]):
    """
    Check if the given cell is upper triangular (if the columns are the lattice vectors).

    Args:
    - cell: 2D list or NumPy array representing the cell

    Returns:
    - True if the cell is upper triangular, False otherwise
    """
    if isinstance(cell,Cell):
        cell = np.asarray(cell).T
        return check_cell_format(cell)
    else:
        n = len(cell)  # Assuming square matrix
        for i in range(n):
            for j in range(0,i):
                if cell[i,j] != 0:
                    return False
        return True
    
#---------------------------------------#
def is_integer(num: Union[int, float, str]) -> bool:
    """
    Check if a given number is an integer.

    Parameters:
        num (Union[int, float, str]): The number to check, which can be an int, float, or string representation of a number.

    Returns:
        bool: True if the number is an integer, False otherwise.
    """
    if isinstance(num, int):
        return True
    elif isinstance(num, float):
        return num.is_integer()
    else:
        try:
            float_num = float(num)
            return float_num.is_integer()
        except ValueError:
            return False
        
# #---------------------------------------#
# def add_info_array(traj:List[Atoms],props:Dict[str,np.ndarray],shapes)->List[Atoms]:
#     new_traj = copy(traj)
#     # shapes = dict()
#     N = len(traj)
#     for k,v in props.items():
#         dtype = shapes[k][0]
#         shape = shapes[k][1]
#         # if 'natoms'
#         if isinstance(shape,int):
#             shape = (N,shape)
#         elif isinstance(shape,tuple):
#             shape = (N,) + shape
#         data = np.reshape(v,shape).astype(dtype)


#     return new_traj
        
#---------------------------------------#
def reshape_info_array(traj:List[Atoms],props:Dict[str,np.ndarray],shapes)->Dict[str,np.ndarray]:
    """
    Reshape arrays in a dictionary according to the given shapes.

    Parameters:
    traj (List[Atoms]): List of ASE Atoms objects representing the trajectory.
    props (Dict[str,np.ndarray]): Dictionary of properties to be reshaped.
    shapes (Dict[str,Tuple[int,...]]): Dictionary of shapes of the properties.

    Returns:
    Dict[str,np.ndarray]: Reshaped dictionary of properties.
    """
    Nconf  = len(traj)
    Natoms = traj[0].get_global_number_of_atoms()
    # print("N conf.  : {:d}".format(Nconf))
    # print("N atoms. : {:d}".format(Natoms))
    whereto = {}
    for k in props.keys():
        if shapes[k][1] is None:
            shapes[k][1] = 1
        # print("reshaping '{:s}' from {}".format(k,data[k].shape),end="")
        if isinstance(shapes[k][1],int) or len(shapes[k][1]) == 1 :
            # info
            whereto[k] = "info"
        elif 'natoms' in shapes[k][1]:
            # arrays
            whereto[k] = "arrays"
        else:
            raise ValueError("coding error")
        
    for k in props.keys():    
        # print("reshaping '{:s}' from {}".format(k,props[k].shape),end="")
        if whereto[k] == "info":
            props[k] = props[k].reshape((Nconf,-1))
        elif whereto[k] == "arrays":
            props[k] = props[k].reshape((Nconf,Natoms,-1))
        else:
            raise ValueError("`whereto[{k}]` can be either `info` or `arrays`.")
        # print(" to {}".format(props[k].shape))
    
    return props, whereto

#---------------------------------------#
def add_info_array(traj:List[Atoms],props:Dict[str,np.ndarray],shapes)->List[Atoms]:
    """
    Add information and/or array attributes to a trajectory.

    Parameters:
    traj (List[Atoms]): List of ASE Atoms objects representing the trajectory.
    props (Dict[str,np.ndarray]): Dictionary of properties to be added.
    shapes (Dict[str,Tuple[int,...]]): Dictionary of shapes of the properties.

    Returns:
    List[Atoms]: Trajectory with the added properties.
    """
    new_traj = copy(traj)
    props, whereto = reshape_info_array(traj,props,shapes)
    # Store data in atoms objects
    for n,atoms in enumerate(new_traj):
        atoms.calc = None  # crucial for writing the info in the file
        for k in props.keys():
            if whereto[k] == "info":
                atoms.info[k] = props[k][n]
            elif whereto[k] == "arrays":
                atoms.arrays[k] = props[k][n]
            else:
                raise ValueError("`whereto[{k}]` can be either `info` or `arrays`.")
    return new_traj

def element_wise_multiplication(A: np.ndarray, B: np.ndarray, axis: int) -> np.ndarray:
    """
    Perform element-wise multiplication of a 1D array with an ND array along a specified axis.

    Parameters:
    A (np.ndarray): 1D array to be multiplied.
    B (np.ndarray): ND array to be multiplied with.
    axis (int): Axis along which to perform the multiplication.

    Returns:
    np.ndarray: Resulting array after element-wise multiplication.

    Raises:
    AssertionError: If A is not a 1D array.

    Examples:
    >>> A = np.array([1, 2, 3])
    >>> B = np.random.rand(2, 3, 4)
    >>> result = element_wise_multiplication(A, B, axis=1)
    >>> result.shape
    (2, 3, 4)

    >>> A = np.array([1, 2])
    >>> B = np.random.rand(2, 3, 2)
    >>> result = element_wise_multiplication(A, B, axis=2)
    >>> result.shape
    (2, 3, 2)
    """
    assert A.ndim == 1, "A must be a 1D array"
    new_shape = [1] * B.ndim
    new_shape[axis] = len(A)
    A_reshaped = A.reshape(new_shape)
    return A_reshaped * B

def is_sorted_ascending(arr: np.ndarray) -> bool:
    """
    Check if the input array is sorted in ascending order.
    
    Args:
        arr (np.ndarray): Input array to be checked.
        
    Returns:
        bool: True if the array is sorted in ascending order, False otherwise.
    """
    return np.all(arr[:-1] <= arr[1:])
def w2_to_w(w2: np.ndarray) -> np.ndarray:
    """
    Compute frequency from frequency squared.

    Args:
        w2 (np.ndarray): Squared frequencies.

    Returns:
        np.ndarray: Corresponding frequencies.
    """
    return np.sqrt(np.absolute(w2)) * np.sign(w2)




def ase2phonopy(atoms_ase: Atoms) -> "PhonopyAtoms":
    """
    Convert ASE Atoms to PhonopyAtoms.

    Args:
        atoms_ase (Atoms): An ASE Atoms object containing the structure data.

    Returns:
        PhonopyAtoms: A PhonopyAtoms object representing the same atomic structure.

    Raises:
        ImportError: If the Phonopy module is not found.
    """
    from phonopy.structure.atoms import PhonopyAtoms
    # Extract the relevant data from ASE Atoms
    cell = atoms_ase.cell  # The cell dimensions of the atomic structure
    positions = atoms_ase.get_positions()  # The positions of atoms in fractional coordinates
    symbols = atoms_ase.get_chemical_symbols()  # The atomic numbers of the atoms
    
    # Create and return PhonopyAtoms
    phonopy_atoms = PhonopyAtoms(
        symbols=symbols,  # Atomic symbols
        positions=positions,  # Atomic positions
        cell=cell,  # The cell dimensions
        pbc=atoms_ase.pbc,  # Periodic boundary conditions
    )
    return phonopy_atoms


def phonopy2ase(atoms_phonopy: "PhonopyAtoms") -> Atoms:
    """
    Convert PhonopyAtoms to ASE Atoms.

    Args:
        atoms_phonopy (PhonopyAtoms): A PhonopyAtoms object containing the structure data.

    Returns:
        Atoms: An ASE Atoms object representing the same atomic structure.

    Raises:
        ImportError: If the ASE module is not found.
    """
    # Create ASE Atoms object from PhonopyAtoms data
    ase_atoms = Atoms(
        cell=atoms_phonopy.cell,  # The cell dimensions of the atomic structure
        scaled_positions=atoms_phonopy.scaled_positions,  # The positions of atoms in fractional coordinates
        numbers=atoms_phonopy.numbers,  # The atomic numbers of the atoms
        pbc=True,  # Set periodic boundary conditions to True
    )
    return ase_atoms
