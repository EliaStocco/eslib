from typing import Callable, Optional, Tuple, Union

import numpy as np

# Documentations;
# - https://stats.stackexchange.com/questions/413209/is-there-something-like-a-root-mean-square-relative-error-rmsre-or-what-is-t

def num_elements_along_axis(arr: np.ndarray, axis: Union[int, Tuple[int]]) -> int:
    """
    Get the number of elements of a numpy array along a specified axis or axes.

    Parameters:
    arr (np.ndarray): The input numpy array.
    axis (Union[int, Tuple[int]]): The axis or tuple of axes along which to count elements.

    Returns:
    int: The number of elements along the specified axis or axes.
    """
    if isinstance(axis, int):
        return arr.shape[axis]
    elif isinstance(axis, tuple):
        return np.prod([arr.shape[ax] for ax in axis])
    else:
        raise ValueError("Axis must be an integer or a tuple of integers.")

#---------------------------------------#
def vectorial_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient for vectorial values.

    The Pearson correlation coefficient is calculated using the formula:
    
    .. math::
        r = \\frac{\\sum_{i=1}^N (x_i - \\bar{x}) \\cdot (y_i - \\bar{y})}
                 {\\sqrt{\\sum_{i=1}^N (x_i - \\bar{x})^2 \\cdot \\sum_{i=1}^N (y_i - \\bar{y})^2}}

    where:
    - x_i and y_i are the vector components of the samples.
    - \\bar{x} and \\bar{y} are the means of the vector components of x and y, respectively.
    - N is the number of samples.

    Parameters:
    x (np.ndarray): A 2D numpy array where each row is a vector representing a sample.
    y (np.ndarray): A 2D numpy array where each row is a vector representing a sample.

    Returns:
    float: The Pearson correlation coefficient between the vectorial values of x and y.

    Raises:
    ValueError: If x and y do not have the same shape.
    AssertionError: If the computed Pearson correlation coefficient is not within the valid range [-1, 1].
    """
    # Check if the input arrays have the same shape
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Calculate the mean of each column (vector component) across all samples
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)

    # Calculate the deviations from the mean
    sigma_x = x - x_mean
    sigma_y = y - y_mean

    # Numerator: sum of element-wise products of deviations
    num = np.sum(sigma_x * sigma_y, axis=1)
    num = np.sum(num)

    # Denominator: product of the sum of squared deviations for x and y
    cov_x = np.sum(np.square(sigma_x))
    cov_y = np.sum(np.square(sigma_y))
    den = np.sqrt(cov_x * cov_y)

    # Pearson correlation coefficient
    r = num / den

    # Check that the result is within the valid range [-1, 1]
    assert np.abs(r) <= 1, "Pearson correlation coefficient out of range [-1, 1]"

    return r

#---------------------------------------#
def p_norm(x: np.ndarray, p: float, axis: Optional[Union[int, Tuple[int]]] = None, keepdims: bool = False, func:Optional[str]="sum") -> np.ndarray:
    """
    Compute the p-norm of a given numpy array along a specified axis.

    Parameters:
    x (np.ndarray): Input array.
    p (float): The order of the norm (p).
    axis (Optional[Union[int, Tuple[int]]]): Axis or axes along which to compute the norm. Default is None.
    keepdims (bool): If True, retains reduced dimensions with size 1. Default is False.

    Returns:
    np.ndarray: The p-norm of the input array along the specified axis.
    """
    if p < 1 and p != np.inf:
        raise ValueError("p must be greater than or equal to 1 or np.inf")
    
    if p == np.inf:
        return np.max(np.abs(x), axis=axis, keepdims=keepdims)
    
    abs_x_p = np.abs(x) ** p
    func = getattr(np,func)
    sum_abs_x_p = func(abs_x_p, axis=axis, keepdims=keepdims)
    return sum_abs_x_p ** (1 / p)

#---------------------------------------#
# Norm
def norm2(x:np.ndarray,**kwargs)->Union[float,np.ndarray]:
    """Norm"""
    return np.sum(np.square(x),**kwargs)

def norm(x:np.ndarray,**kwargs)->Union[float,np.ndarray]:
    """Norm"""
    x = norm2(x=x,**kwargs)
    out = np.sqrt(x)
    assert np.allclose(np.linalg.norm(x,**kwargs),out)
    return out

def RMS(x:np.ndarray,axis:Union[Tuple[int],int],**kwargs)->np.ndarray:
    """Root Mean Square"""
    return p_norm(x=x,p=2,axis=axis,func="mean")

#---------------------------------------#
# Error/distances/loss functions

def L2_norm(x: np.ndarray,**kwargs)->Union[float,np.ndarray]:
    se   = np.square(x)         #           squared error
    mse  = np.mean(se,**kwargs) #      mean squared error
    rmse = np.sqrt(mse)         # root mean squared error
    return rmse

def MSE(pred:np.ndarray,ref:np.ndarray,**kwargs)->Union[float,np.ndarray]:
    """Mean Squared Error"""
    err  = pred - ref           #              error
    se   = np.square(err)       #      squared error
    mse  = np.mean(se,**kwargs) # mean squared error
    return mse

def RMSE(pred:np.ndarray,ref:np.ndarray,**kwargs)->Union[float,np.ndarray]:
    """Root Mean Squared Error"""
    err  = pred - ref           # error
    return L2_norm(err)

def RMSRE(pred:np.ndarray,ref:np.ndarray,**kwargs)->Union[float,np.ndarray]:
    """Root Mean Squared Relative Error"""
    err   = pred - ref            #                            error
    rel   = err / ref             #                   relative error
    return L2_norm(rel)

def RRMSE(pred:np.ndarray,ref:np.ndarray,**kwargs)->Union[float,np.ndarray]:
    """Relative Root Mean Squared Error"""
    err   = pred - ref                 #                            error
    se    = np.square(err)             #                    squared error
    mse   = np.mean(se,**kwargs)       #               mean squared error
    rmse  = mse/np.sum(np.square(ref)) #      relative mean squared error
    rrmse = np.sqrt(rmse)              # root relative mean squared error
    return rrmse

#---------------------------------------#
# Elia Stocco's custom functions for atomic references

def MAR(x:np.ndarray)->np.ndarray:
    """Mean Atomic Reference"""
    assert x.ndim == 3
    # reshape array
    dim = int(np.sqrt(x.shape[2]))
    shape = (x.shape[0],x.shape[1],dim,dim)
    ref = x.reshape(shape)
    # compute the trace/dim along the last 2 axes
    ref = np.mean(ref,axis=(2,3),keepdims=False)
    # reshape back to the original shape
    ref = np.repeat(ref, x.shape[2])
    ref = ref.reshape(x.shape)
    return ref

def RMSAR(x:np.ndarray)->np.ndarray:
    """Root Mean Squared Atomic Reference"""
    assert x.ndim == 3
    # reshape array
    dim = int(np.sqrt(x.shape[2]))
    shape = (x.shape[0],x.shape[1],dim,dim)
    ref = x.reshape(shape)
    # compute the trace/dim along the last 2 axes
    ref = np.square(ref)
    ref = np.mean(ref,axis=(2,3),keepdims=False)
    ref = np.sqrt(ref)
    # reshape back to the original shape
    ref = np.repeat(ref, x.shape[2])
    ref = ref.reshape(x.shape)
    return ref

def MDAR(x:np.ndarray)->np.ndarray:
    """Mean Diagonal Atomic Reference"""
    assert x.ndim == 3
    # reshape array
    dim = int(np.sqrt(x.shape[2]))
    shape = (x.shape[0],x.shape[1],dim,dim)
    ref = x.reshape(shape)
    # compute the trace/dim along the last 2 axes
    ref = np.diagonal(ref, axis1=2, axis2=3)
    ref = np.mean(ref,axis=2,keepdims=False)
    # reshape back to the original shape
    ref = np.repeat(ref, x.shape[2])
    ref = ref.reshape(x.shape)
    return ref

def RMSDAR(x:np.ndarray)->np.ndarray:
    """Root Mean Squared Diagonal Atomic Reference"""
    assert x.ndim == 3
    # reshape array
    dim = int(np.sqrt(x.shape[2]))
    shape = (x.shape[0],x.shape[1],dim,dim)
    ref = x.reshape(shape)
    # compute the trace/dim along the last 2 axes
    ref = np.diagonal(ref, axis1=2, axis2=3)
    ref = np.square(ref)
    ref = np.mean(ref,axis=2,keepdims=False)
    ref = np.sqrt(ref)
    # reshape back to the original shape
    ref = np.repeat(ref, x.shape[2])
    ref = ref.reshape(x.shape)
    return ref

#---------------------------------------#
# Elia Stocco's custom functions

#------------------------------#
def RMSRAE(pred:np.ndarray,ref:np.ndarray,func:Callable[[np.ndarray], np.ndarray]=MDAR,**kwargs)->np.ndarray:
    """Root Mean Squared Relative Atomic Error"""
    ar    = func(ref)             #                        atomic reference
    err   = pred - ref            #                                   error
    rel   = err / ar              #                   relative atomic error
    sre   = np.square(rel)        #           squared relative atomic error
    msre  = np.mean(sre,**kwargs) #      mean squared relative atomic error
    rmsre = np.sqrt(msre)         # root mean squared relative atomic error
    return rmsre

#------------------------------#
# Functions derived from RMSRAE

def RMSRAE_MAR(**kwargs)->np.ndarray:
    """Root Mean Squared Relative Atomic Error, using Mean Atomic Reference."""
    return RMSRAE(**kwargs,func=MAR)

def RMSRAE_RMSAR(**kwargs)->np.ndarray:
    """Root Mean Squared Relative Atomic Error, using Root Mean Squared Atomic Reference."""
    return RMSRAE(**kwargs,func=RMSAR)

def RMSRAE_MDAR(**kwargs)->np.ndarray:
    """Root Mean Squared Relative Atomic Error, using Mean Diagonal Atomic Reference."""
    return RMSRAE(**kwargs,func=MDAR)

def RMSRAE_RMSDAR(**kwargs)->np.ndarray:
    """Root Mean Squared Relative Atomic Error, using Root Mean Squared Diagonal Atomic Reference."""
    return RMSRAE(**kwargs,func=RMSDAR)

#------------------------------#
def RRAMSE(pred:np.ndarray,ref:np.ndarray,func:Callable[[np.ndarray], np.ndarray]=MDAR,**kwargs)->np.ndarray:
    """Root Relative Atomic Mean Squared Error"""
    ar     = func(ref)                 #                        atomic reference
    err    = pred - ref                #                                   error
    se     = np.square(err)            #                           squared error
    mse    = np.mean(se,**kwargs)      #                      mean squared error
    ramse  = mse/np.sum(np.square(ar)) #      relative atomic mean squared error
    rramse = np.sqrt(ramse)            # root relative atomic mean squared error
    return rramse

#------------------------------#
# Functions derived from RRAMSE
def RRAMSE_MAR(**kwargs)->np.ndarray:
    """Root Relative Atomic Mean Squared Error, using Mean Atomic Reference."""
    return RRAMSE(**kwargs,func=MAR)

def RRAMSE_RMSAR(**kwargs)->np.ndarray:
    """Root Relative Atomic Mean Squared Error, using Root Mean Squared Atomic Reference."""
    return RRAMSE(**kwargs,func=RMSAR)

def RRAMSE_MDAR(**kwargs)->np.ndarray:
    """Root Relative Atomic Mean Squared Error, using Mean Diagonal Atomic Reference."""
    return RRAMSE(**kwargs,func=MDAR)

def RRAMSE_RMSDAR(**kwargs)->np.ndarray:
    """Root Relative Atomic Mean Squared Error, using Root Mean Squared Diagonal Atomic Reference."""
    return RRAMSE(**kwargs,func=RMSDAR)

#---------------------------------------#
# Dictionary of regression metrics
metrics = {
    "mse"     : MSE,
    "rmse"    : RMSE,
    "rrmse"   : RRMSE,
    "rmsre"   : RMSRE,
    "rmsrae_mar"    : RMSRAE_MAR,
    "rmsrae_rmsar"  : RMSRAE_RMSAR,
    "rmsrae_mdar"   : RMSRAE_MDAR,
    "rmsrae_rmsdar" : RMSRAE_RMSDAR,
    "rramse_mar"    : RRAMSE_MAR,
    "rramse_rmsar"  : RRAMSE_RMSAR,
    "rramse_mdar"   : RRAMSE_MDAR,
    "rramse_rmsdar" : RRAMSE_RMSDAR,
    "vecr" : lambda x,y : vectorial_pearson(x,y)
}

def RMSEforces(pred:np.ndarray,ref:np.ndarray):
    """Return the RMSE of the forces for a bunch of atomic structures.
    It does not perform the mean over the structures.
    """
    assert pred.ndim == 3, "error"
    assert pred.shape[1:] == ref.shape[1:], "error"
    
    delta = np.square(pred-ref).sum(axis=2) # su over x,y,z
    err = np.mean(delta,axis=1) # mean over atoms
    return np.sqrt(err)
    