from typing import Optional
import numpy as np
import pandas as pd
from typing import Tuple
from warnings import warn

def levi_civita():
    """Returns the 3x3x3 Levi-Civita tensor."""
    tensor = np.zeros((3, 3, 3), dtype=int)
    indices = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    
    for i, j, k in indices:
        tensor[i, j, k] = (1 if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                            else -1 if (i, j, k) in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]
                            else 0)
    return tensor
    

def mean_std_err(array: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(array, axis=axis)
    std, err = std_err(array, axis=axis)
    return mean, std, err

def mean_std_err2pandas(array: np.ndarray, axis: int)->pd.DataFrame:
    mean, std, err = mean_std_err(array,axis)
    return pd.DataFrame({"mean":mean,"std":std,"err":err})
    
def std_err(array: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the standard deviation and standard error along a specified axis.

    Parameters:
    array : np.ndarray
        Input data array.
    axis : int
        Axis along which computations are performed.

    Returns:
    Tuple[np.ndarray, np.ndarray]
        Standard deviation and standard error along the specified axis.

    Notes:
    - Requires at least two elements along the axis to compute the error.
    """
    # Compute standard deviation along the specified axis
    std = np.std(array, axis=axis, ddof=0)

    # Compute standard error of the mean along the specified axis
    n = array.shape[axis]
    if n <= 1:
        warn("Standard error requires at least two elements along the axis.")
        err = np.full_like(std,np.nan)
    else:
        err = std / np.sqrt(n - 1)

    return std, err


def reshape_into_blocks(data: np.ndarray, N: int) -> np.ndarray:
    """
    Reshape a numpy array into N blocks along the first axis, discarding any excess elements.

    Parameters:
        data (numpy.ndarray): The input array to be reshaped.
        N (int): The number of blocks to reshape the data into.

    Returns:
        numpy.ndarray: The reshaped array containing N blocks along the first axis.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> N = 3
        >>> reshaped_data = reshape_into_blocks(data, N)
        >>> print(reshaped_data)
        [[1 2]
         [3 4]
         [5 6]]

        >>> data = np.arange(24).reshape(6, 4)
        >>> N = 3
        >>> reshaped_data = reshape_into_blocks(data, N)
        >>> print(reshaped_data)
        [[[ 0  1  2  3]
          [ 4  5  6  7]]
         [[ 8  9 10 11]
          [12 13 14 15]]
         [[16 17 18 19]
          [20 21 22 23]]]

        >>> data = np.arange(36).reshape(6, 3, 2)
        >>> N = 2
        >>> reshaped_data = reshape_into_blocks(data, N)
        >>> print(reshaped_data)
        [[[[ 0  1]
           [ 2  3]
           [ 4  5]]

          [[ 6  7]
           [ 8  9]
           [10 11]]]


         [[[12 13]
           [14 15]
           [16 17]]

          [[18 19]
           [20 21]
           [22 23]]]]
    """
    # Calculate the number of elements per block along the first axis
    elements_per_block = data.shape[0] // N

    # Truncate data along the first axis to a length divisible by N
    truncated_length = elements_per_block * N
    truncated_data = data[:truncated_length]

    # Reshape the truncated data into N blocks along the first axis
    new_shape = (N, elements_per_block) + data.shape[1:]
    reshaped_data = truncated_data.reshape(new_shape)

    return reshaped_data



def tacf(data:np.ndarray)->np.ndarray:
    """
    Compute the Time AutoCorrelation Function
    """
    fft = np.fft.rfft(data,axis=0)
    ft_ac = fft * np.conjugate(fft)
    autocorr = np.fft.irfft(ft_ac,axis=0)[:int(int(len(data)/2)+1)]
    autocorr /= np.mean(data**2,axis=0)*len(data)
    return autocorr

def histogram_along_axis(data: np.ndarray, bins: int, axis: int) -> np.ndarray:
    """
    Compute the histogram of a numpy array along a specific axis.

    Parameters:
        data (numpy.ndarray): The input array.
        bins (int): The number of bins for the histogram.
        axis (int): The axis along which to compute the histogram.

    Returns:
        numpy.ndarray: The histogram values along the specified axis.

    Example:
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> bins = 5
        >>> axis = 1
        >>> hist = histogram_along_axis(data, bins, axis)
        >>> print(hist)
        array([[1, 1, 1, 0, 0],
               [1, 1, 1, 0, 0],
               [1, 1, 1, 0, 0]])
    """
    # Transpose the array to move the specified axis to the first position
    data_transposed = np.transpose(data, np.roll(np.arange(data.ndim), -axis))

    # Compute the histogram along the first axis
    range = (np.min(data), np.max(data))
    hist = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, range=range)[0], 0, data_transposed)

    # Transpose the result back to the original shape
    hist_transposed = np.transpose(hist, np.roll(np.arange(hist.ndim), axis))

    return hist_transposed

def cumulative_mean(x:np.ndarray,axis:Optional[int]=0)->np.ndarray:
    """
    Compute the cumulative mean of an array along a specified axis.

    Parameters:
        x (np.ndarray): Input array for which to compute the cumulative mean.
        axis (int, optional): The axis along which to compute the cumulative mean. Default is 0.

    Returns:
        np.ndarray: An array of the same shape as `x` with the cumulative means computed along the specified axis.
    """

    return np.cumsum(x,axis=axis)/np.arange(1,x.shape[axis]+1)

def melt(A: np.ndarray, name: str = "value") -> pd.DataFrame:
    """
    R-like function to melt a numpy array into a long-format pandas DataFrame.
    Returns a DataFrame with integer index columns and a value column matching A's dtype.
    """
    shape = A.shape
    ndim = len(shape)
    indices = np.indices(shape).reshape(ndim, -1)  # shape: (n_dims, n_elements)
    values = A.flatten()  # flatten A
    
    # Prepare dictionary with correct types directly
    data = {
        f"dim_{i}": indices[i].astype(np.int32)  # or int64 if preferred
        for i in range(ndim)
    }
    data[name] = values  # inherits dtype from A

    return pd.DataFrame(data)


# def aggregate_by_group(A: np.ndarray, B: np.ndarray, axis: int, aggregation_function=np.sum) -> np.ndarray:
#     """
#     General function to aggregate values of A along the specified axis based on 
#     the grouping indices in B.
    
#     A: numpy array of shape (n_structures, ..., n_atoms, n_features) 
#        where ... represents other dimensions (if applicable).
#     B: numpy array of shape (n_structures, ..., n_atoms) with integer grouping indices (e.g., unit-cell indices).
#     axis: the axis along which the aggregation happens (e.g., atoms dimension).
#     aggregation_function: function (default: np.sum) to apply to the grouped elements.
    
#     Returns:
#     aggregated_A: numpy array of shape (n_structures, ..., n_groups, n_features),
#                   where each group is aggregated along the specified axis.
#     """
    
#     A_tmp = transform_to_indexed_values(A)
#     # Extract the shape information for generalization
#     shape = A.shape
#     n_structures = shape[0]
#     n_atoms = shape[axis]  # size along the specified axis (e.g., atoms)
#     n_features = shape[-1]  # size along the features axis
    
#     # Initialize the list to store the aggregated results
#     aggregated_A = []
    
#     for i in range(n_structures):
#         # Initialize an empty list to store aggregated results for this structure
#         structure_aggregated = []
        
#         # Get the unique group indices for the current structure (along the atoms dimension)
#         unique_groups = np.unique(B[i])
        
#         for group in unique_groups:
#             # Find all the elements in A that belong to the current group
#             group_indices = (B[i] == group)
#             group_elements = A[i][group_indices]
            
#             # Apply the aggregation function to the atoms in the current group along the specified axis
#             group_aggregated = aggregation_function(group_elements, axis=axis)
            
#             # Append the aggregated group result
#             structure_aggregated.append(group_aggregated)
        
#         # Append the aggregated results for the current structure
#         aggregated_A.append(structure_aggregated)
    
#     # Convert the list of results into a numpy array and return
#     return np.array(aggregated_A)


