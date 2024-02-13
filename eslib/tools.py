import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.geometry import distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Union

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
    from ipi.utils.units import unit_to_internal, unit_to_user
    if family is not None:
        factor = unit_to_internal(family, _from, 1)
        factor *= unit_to_user(family, _to, 1)
        return what * factor
    else :
        return what

#---------------------------------------#
# Decorator to convert ase.Cell to np.array and transpose
def ase_cell_to_np_transpose(func):
    """Decorator to convert an ASE cell to NumPy array and transpose for use in the decorated function."""
    def wrapper(cell, *args, **kwargs):
        if isinstance(cell, Cell):
            cell = np.asarray(cell).T
        return func(cell, *args, **kwargs)
    return wrapper
#---------------------------------------#
def return_transformed_components(func):
    """Decorator to automatically compute the matrix multiplication if a vector is provided."""
    def wrapper(cell:Union[np.ndarray,Cell],v:np.ndarray=None, *args, **kwargs):
        matrix = func(cell=cell,v=None,*args, **kwargs)
        if v is None:
            return matrix
        else:
            v = np.asarray(v).reshape((-1,3))
            return (matrix @ v.T).T
    return wrapper

#---------------------------------------# 
@return_transformed_components
@ase_cell_to_np_transpose
def cart2lattice(cell:Union[np.ndarray,Cell],v:np.ndarray=None): #,*argc,**argv):
    """ Cartesian to lattice coordinates rotation matrix
    
    Input:
        cell: lattice parameters, 
            where the i^th basis vector is stored in the i^th columns
            (it's the opposite of ASE, QE, FHI-aims)
            lattice : 
                | a_1x a_2x a_3x |
                | a_1y a_2y a_3y |
                | a_1z a_2z a_3z |
    Output:
        rotation matrix
    """
    matrix = lattice2cart(cell)
    matrix = np.linalg.inv( matrix )
    return matrix

#---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def lattice2cart(cell:Union[np.ndarray,Cell],v:np.ndarray=None): #,*argc,**argv):
    """ Lattice to Cartesian coordinates rotation matrix
    
    Input:
        cell: lattice parameters, 
            where the i^th basis vector is stored in the i^th columns
            (it's the opposite of ASE, QE, FHI-aims)
            lattice : 
                | a_1x a_2x a_3x |\n
                | a_1y a_2y a_3y |\n
                | a_1z a_2z a_3z |\n
    Output:
        rotation matrix
    """

    if cell.shape != (3,3):
        raise ValueError("lattice with wrong shape:",cell.shape)
    from copy import copy
    # I have to divide normalize the lattice parameters
    length = np.linalg.norm(cell,axis=0)
    matrix = copy(cell)
    # normalize the columns
    for i in range(3):
        matrix[:,i] /= length[i]
    return matrix

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