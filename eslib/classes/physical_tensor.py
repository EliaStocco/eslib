import xarray as xr
import numpy as np
from eslib.units import *
from typing import TypeVar
T = TypeVar('T', bound='PhysicalTensor')

threshold = 1e-18

def all_positive(x):
    return np.any(np.logical_and( remove_unit(x)[0] < threshold , np.abs(remove_unit(x)[0]) > threshold ))

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

    
class PhysicalTensor(xr.DataArray):
    __slots__ = ()
    data:np.ndarray

    def cdot(self:T,B:T,dim:str)->T:
        return dot(self,B,dim)