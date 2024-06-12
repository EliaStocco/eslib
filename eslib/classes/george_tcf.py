import numpy as np
from typing import Optional

from __future__ import absolute_import, division, print_function
import sys

def slice_along_axis(arr, axis, start=None, end=None, step=1):
    """Return arr[..., slc, ...], where the slice is applied to 
    a specified axis

    Args:
        arr (np.ndarray)
        axis (int)
        start (int, optional): Defaults to None.
        end (int, optional): Defaults to None.
        step (int, optional): Defaults to 1.
    """
    return arr[ (axis % arr.ndim)*(slice(None),) + (slice(start, end, step),)]

def idx_along_axis(arr, axis, idx):
    """Return arr[..., idx, ...], where the idx refers to the specified axis

    Args:
        arr : np.ndarray
        axis : int
        idx : int
    """
    return arr[ (axis % arr.ndim)*(slice(None),) + (idx,)]

def append_dims(arr, ndims=1):
    """Return a view of the input array with `ndims` axes of
    size one appended.
    """
    return arr[(Ellipsis,) + ndims*(None,)] 



def index_in_slice(slice_obj: slice, index: int) -> bool:
    """
    Check if the given index is within the specified slice object.

    Parameters:
    slice_obj (slice): The slice object to check against.
    index (int): The index to check.

    Returns:
    bool: True if the index is within the slice, False otherwise.
    """
    # Create a range object from the slice object's start, stop, and step attributes
    range_obj = range(slice_obj.start if slice_obj.start else 0,
                      slice_obj.stop if slice_obj.stop else sys.maxsize,
                      slice_obj.step if slice_obj.step else 1)

    # Check if the index is within the range object
    return index in range_obj

def correlate(
        A: np.ndarray, 
        B: np.ndarray, 
        axis: Optional[int] = 0) -> np.ndarray:
    
    assert A.ndim == B.ndim
    shape_a = A.shape
    len_a = shape_a[axis]
    shape_b = B.shape
    len_b = shape_b[axis]
    
    if len_a >= len_b:
        a_wkspace = slice_along_axis(A, axis, end=len_b)
        len_fft = 2*len_b
        len_tcf = len_b
        norm_tcf = np.arange(len_b, 0, -1, dtype=int)
    else:
        len_tcf = len_b
        len_fft = len_a + len_b
        a_wkspace = A
        norm_tcf = np.arange(len_b, 0, -1, dtype=int)
        norm_tcf = np.where(norm_tcf > len_a, len_a, norm_tcf)

    dims_to_append = np.arange(B.ndim-1, -1, -1, dtype=int)[axis]
    norm_tcf = append_dims(norm_tcf, dims_to_append)
    ftA = np.fft.rfft(a_wkspace, axis=axis, n=len_fft)
    ftB = np.fft.rfft(B, axis=axis, n=len_fft)
    np.conj(ftA, out=ftA)
    ftB *= ftA
    out = np.fft.irfft(ftB, axis=axis, n=len_fft)[:len_tcf] / norm_tcf
    return out