import numpy as np
from typing import Optional

def padding(array:np.ndarray,pad:int,axis:Optional[int]=0)->np.ndarray:
    assert axis == 0, "not implemented yet"
    # N = array.shape[axis]*pad
    # padding = np.zeros(N)
    
    shape = list(array.shape)
    shape[axis] += pad
    shape = tuple(shape)
    
    padded = np.zeros(shape,dtype=array.dtype)
    array = np.moveaxis(array,axis,0)
    padded[:array.shape[0], ... ] = array
    # padded[array.shape[0]:, ... ] = 0
    
    padded = np.moveaxis(padded,0,axis)
    return padded

    # return np.asarray([ np.append(a,padding) for a in array ])