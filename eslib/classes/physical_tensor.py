import xarray as xr
import numpy as np
import glob
from eslib.formatting import float_format, complex_format
from eslib.classes.io import pickleIO
from eslib.units import *
from typing import TypeVar, Union, Any, Callable, Type
T = TypeVar('T', bound='PhysicalTensor')

threshold = 1e-18

def all_positive(x):
    return np.any(np.logical_or( remove_unit(x)[0] < threshold , np.abs(remove_unit(x)[0]) > threshold ))

def corrected_sqrt(x):
    return np.sqrt(np.abs(x)) * np.sign(x)

def inv(A:xr.DataArray)->xr.DataArray:
    """Calculate the inverse of a 2D ```xarray.DataArray``` using ```np.linalg.inv``` while preserving the xarray structure."""
    _A, unit = remove_unit(A)    
    if _A.ndim != 2:
        raise ValueError("Input DataArray must be 2D.")
    # Calculate the inverse of the 2D array
    inv_data = np.linalg.inv(_A)
    # Create a new DataArray with the inverted values and the original coordinates
    inv_da = xr.DataArray(inv_data.T, dims=_A.dims, coords=_A.coords)
    return set_unit(inv_da,1/unit) 

def rbc(A:xr.DataArray,B:xr.DataArray,dim:str):
    """Row by column multiplication between two ```xarray.DataArray``` ```A``` and ```B``` along the specified dimension ```dim```"""
    # Check if A and B have at least two dimensions with the same name, and one of them is 'dim'
    common_dims = set(A.dims).intersection(B.dims)
    if len(common_dims) < 2 or dim not in common_dims:
        raise ValueError("Both input arrays must have at least two dimensions with the same name, and one of them must be the specified 'dim'.")
    # Determine the common dimension that is not 'dim'
    other_common_dim = next(d for d in common_dims if d != dim)
    # Rename the common dimension for A and B
    _A = A.rename({other_common_dim: f'{other_common_dim}-left'})
    _B = B.rename({other_common_dim: f'{other_common_dim}-right'})
    # compute
    _A_,ua = remove_unit(_A)
    _B_,ub = remove_unit(_B)
    out = dot(_A_,_B_,dim=dim)
    return set_unit(out,ua*ub)

def dot(A:xr.DataArray,B:xr.DataArray,dim:str):
    """Dot product (contraction) between two ```xarray.DataArray``` ```A``` and ```B``` along the specified dimension ```dim```"""
    _A,ua = remove_unit(A)
    _B,ub = remove_unit(B)
    out = _A.dot(_B,dim=dim)
    newu = ua*ub
    return set_unit(out,newu)

def norm_by(array,dim):
    tmp = np.linalg.norm(array.data,axis=array.dims.index(dim))
    tmp, _ = remove_unit(tmp)
    return tmp * get_unit(array)

def diag_matrix(M,exp):
    out = np.eye(len(M))        
    if exp == "-1":
        np.fill_diagonal(out,1.0/M)
    elif exp == "1/2":
        np.fill_diagonal(out,np.sqrt(M))
    elif exp == "-1/2":
        np.fill_diagonal(out,1.0/np.sqrt(M))
    else :
        raise ValueError("'exp' value not allowed")
    return out  

# def cast_to_parent(obj,basecls):
#     """Cast a child class object to the parent class object without copying all the attributes."""
#     obj.__class__ = basecls
#     obj.__name__ = basecls.__name__
#     return obj

def read_from_pattern(func: Callable) -> Callable:
    """
    Decorator to handle file patterns and apply a function to all matching files.

    This decorator checks if the provided file name contains wildcard characters.
    If it does, it iterates over all matching files and applies the decorated function
    to each file, returning a list of results. If it doesn't, it applies the function
    directly to the provided file.

    Parameters:
    func (Callable): The function to be decorated.

    Returns:
    Callable: The wrapped function that handles file patterns.

    Raises:
    ValueError: If no file is specified in the arguments.

    Examples:
    >>> class Example:
    >>>     @read_from_pattern
    >>>     def load(cls, file: str):
    >>>         return f"Loading {file}"

    >>> example = Example()
    >>> results = example.load(file="*.txt")
    >>> isinstance(results, list)
    True
    >>> all(isinstance(res, str) for res in results)
    True

    >>> result = example.load(file="example.txt")
    >>> isinstance(result, str)
    True
    """
    def wrapper(cls: Type, **argv: Any) -> Any:
        file = argv.get('file') or argv.get('fname')
        if not file:
            raise ValueError("File not specified")
        
        if any(char in file for char in '*?[]'):
            # If the file is a pattern, iterate over all matching files
            all_files = glob.glob(file)
            results = [None] * len(all_files)
            for n, matched_file in enumerate(all_files):
                new_argv = argv.copy()
                new_argv['file'] = matched_file
                results[n] = func(cls, **new_argv)
            return cls(results)
        else:
            # If the file is not a pattern, call the function directly
            return func(cls, **argv)
    return wrapper

class PhysicalTensor(pickleIO,xr.DataArray):
    # __slots__ = ()
    # data:np.ndarray

    def cdot(self:T,B:T,dim:str)->T:
        return dot(self,B,dim)
    
    def to_data(self:T)->np.ndarray:
        return remove_unit(self)[0].to_numpy()
    
    @classmethod
    @read_from_pattern
    @pickleIO.correct_extension_in
    def from_file(cls, **argv):
        """
        Load atomic structures from file.

        Attention: it's recommended to use keyword-only arguments.
        """
        if 'file' in argv: 
            file = str(argv['file'])
            del argv['file']
        elif 'fname' in argv:
            file = str(argv['fname'])
            del argv['fname']
        else:
            raise ValueError("error")
        if file.endswith("txt"):
            data = np.loadtxt(fname=file,**argv)
        elif file.endswith("npy"):
            data = np.load(file,**argv)
        else:
            raise ValueError("Only `txt`, `pickle`, and `npy` extensions are supported.")
        return cls(data)
    
    @pickleIO.correct_extension_out
    def to_file(self: T, file: str,fmt:str=float_format,**argv):
        """
        Write atomic structures to file.
        
        Attention: it's recommended to use keyword-only arguments.
        """
        data = self.to_data()
        if file.endswith("txt"):
            # fmt = float_format  else complex_format
            if not np.any(np.iscomplex(data)):
                np.savetxt(file,data,fmt=fmt,**argv) # fmt)
            else:
                np.savetxt(file,data,**argv)
        elif file.endswith("npy"):
            np.save(file,data)
        else:
            raise ValueError("Only `txt`, `pickle`, and `npy` extensions are supported.")