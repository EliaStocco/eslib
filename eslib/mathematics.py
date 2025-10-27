from typing import Optional
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from warnings import warn

from eslib.functional import extend2NDarray

def levi_civita():
    """Returns the 3x3x3 Levi-Civita tensor."""
    tensor = np.zeros((3, 3, 3), dtype=int)
    indices = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    
    for i, j, k in indices:
        tensor[i, j, k] = (1 if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                            else -1 if (i, j, k) in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]
                            else 0)
    return tensor
    

def mean_std_err(array: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def pandas2ndarray(
    df: pd.DataFrame,
    index_columns: List[str],
    ignore_columns: List[str] = []
) -> Tuple[np.ndarray, Dict[str, List[Any]]]:
    """
    Converts a DataFrame into an N-dimensional NumPy array and returns axis interpretation.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        index_columns (List[str]): Columns used as index dimensions (int or str).
        ignore_columns (List[str]): Columns to ignore when selecting value columns.

    Returns:
        Tuple:
            - np.ndarray: The resulting array.
            - Dict: Mapping of axis names (e.g., 'region_axis_0') to original labels used as indices.
    """
    _df = df.copy()
    axis_info: Dict[str, List[Any]] = {}

    # Map categorical/string index columns to integer codes
    for axis_idx, col in enumerate(index_columns):
        dtype = _df[col].dtype
        if np.issubdtype(dtype, np.str_):
            cats = pd.Categorical(_df[col])
            _df[col] = cats.codes
            axis_info[f'{col}_axis_{axis_idx}'] = cats.categories.tolist()
        elif np.issubdtype(dtype, np.integer):
            unique_vals = sorted(_df[col].unique())
            val_to_idx = {val: i for i, val in enumerate(unique_vals)}
            _df[col] = _df[col].map(val_to_idx)
            axis_info[f'{col}'] = unique_vals
        else:
            raise TypeError(f"Index column '{col}' must be int or str, got {dtype}")

    # Determine value columns
    value_columns = [c for c in _df.columns if c not in index_columns and c not in ignore_columns]
    values = np.stack(_df[value_columns].to_numpy())

    # Create output array
    shape = [_df[col].max() + 1 for col in index_columns] + [len(value_columns)]
    result = np.full(shape, np.nan, dtype=float)

    # Fill values
    idx = tuple(_df[col].to_numpy() for col in index_columns)
    result[idx] = values
    
    for n,(_,r) in enumerate(axis_info.items()):
        assert result.shape[n] == len(r), f"axis {n} has length {result.shape[n]} but {len(r)} values"

    return result, {**axis_info,"values":value_columns}

def dcast(df:pd.DataFrame, index_columns:List[str], ignore_columns:List[str]=[])->np.ndarray:
    """
    R-like dcast function that reshapes a pandas DataFrame into a 3D NumPy array.
    
    This function mimics the behavior of R's `dcast` function. It transforms long-format
    data (with index columns and value columns) into a 3D array, where the first two 
    dimensions correspond to the index columns, and the third dimension holds the values 
    from the remaining columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with index columns and value columns.
        index_columns (list): List of column names to be used as indices for the resulting array.
        ignore_columns (list): List of column names to ignore (these will not be included in the values).
        
    Returns:
        np.ndarray: 3D NumPy array, where indices correspond to values in the index columns
                    and values correspond to the remaining columns.
    """
    # Ensure correct data types for index columns (integer or category)
    df[index_columns] = df[index_columns].astype(int)

    # Determine the value columns by excluding index_columns and ignore_columns
    value_columns = [col for col in df.columns if col not in index_columns and col not in ignore_columns]

    # Combine the value columns into a single list column
    NAME = "values"
    if NAME in df.columns:
        NAME = "_values_"
    
    df[NAME] = df[value_columns].values.tolist()

    # Pivot the DataFrame: this will arrange data such that indices form the axes
    pivot = df.pivot(index=index_columns[0], columns=index_columns[1], values=NAME)

    # Convert the pivoted DataFrame into a 3D NumPy array
    result_array = np.array(pivot.values.tolist())
    
    del df[NAME]  # Clean up the temporary column

    return result_array

def get_indices(shapes:List[int])->np.ndarray:
    return np.indices(shapes).reshape(len(shapes),-1).T

def flatten_except(A: np.ndarray, value_axes: list) -> np.ndarray:
    """
    Flatten all dimensions of a numpy array except for those specified in value_axes.

    Parameters:
        A (np.ndarray): The input N-dimensional array.
        value_axes (list): List of axes that should be preserved (not flattened).

    Returns:
        np.ndarray: The reshaped (flattened) array.
    """
    # Get the shape of the input array
    shape = A.shape
    ndim = A.ndim
    
    # Validate that the value_axes are within the valid range
    assert all(axis < ndim for axis in value_axes), "All value_axes must be less than the number of dimensions"
    
    # Determine the axes that will be flattened
    flatten_axes = [i for i in range(ndim) if i not in value_axes]
    
    # Calculate the new shape: Keep value_axes dimensions and flatten the others
    new_shape = tuple(shape[axis] for axis in value_axes) + (-1,)  # Flatten other axes
    reshaped_array = A.reshape(new_shape)
    
    return reshaped_array

def melt(
    A: np.ndarray,
    index: Dict[int, str],  # Use a dictionary for index axes with names
    value_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    R-like function to melt a numpy array into a long-format pandas DataFrame with support for multiple value columns.

    Parameters:
        A (np.ndarray): Input N-dimensional array.
        index (Dict[int, str]): Dictionary where keys are the axis indices and values are the column names.
        value_names (List[str], optional): List of names for the value columns. If None, the default names are used as "value_1", "value_2", ...

    Returns:
        pd.DataFrame: A long-format DataFrame with index columns and value columns.
    """
    # Ensure A is a NumPy array
    A = np.asarray(A)
    
    # Get the shape of the N-dimensional array
    shape = A.shape
    ndim = A.ndim

    # Extract the indices for the index columns from the provided dictionary
    index_axes = list(index.keys())
    
    
    # Determine value axes by excluding index axes
    value_axes = [i for i in range(ndim) if i not in index_axes]
    if len(value_axes) == 0:
        A = A[...,np.newaxis]  # Add a new axis if no value axes are found
        return melt(A, index, value_names)
    
    assert len(value_axes) == 1, "There should be exactly one value axis."
    value_axes = value_axes[0]  # Get the single value axis index
    
    # If value names are not provided, create default names for the value columns
    assert value_names is not None, "error"
    # value_names = [f"value_{i+1}" for i in range(len(value_axes))]
    
    # Generate meshgrid for all indices in the array
    ii = np.asarray(shape)[index_axes]
    assert len(ii) == A.ndim-1 , "Mismatch between index axes and array dimensions."
    indices = get_indices(ii)  # shape: (n_elements, n_dims)
    
    assert indices.ndim == 2, "Indices should be 2D after reshaping."
    assert indices.shape[0] == np.prod(ii), "Indices shape mismatch with the product of index axes."
    assert indices.shape[1] == len(index_axes), "Indices shape mismatch with index axes."
    
    # Create a dictionary to hold the reshaped data
    data = {}
    
    for i, (axis, name) in enumerate(index.items()):
        tmp = indices[:, i].astype(np.int32)  
        data[name] = tmp  # Convert to int32 for index columns
    
    A = np.moveaxis(A,value_axes,-1)
    A = A.reshape((-1,A.shape[-1]))
    assert A.ndim == 2, "Reshaped array should be 2D after moving axes."
    assert A.shape[0] == np.prod(ii), "Reshaped array shape mismatch with the product of original shape."
    
    for n in range(A.shape[1]):
        data[value_names[n]] = A[:,n]
    
    df = pd.DataFrame(data)
    
    # Return DataFrame
    return df

def centered_window(window_name: str, cf_len: int, window_width: int) -> np.ndarray:
    """
    Create a zero-padded, centered window over cf_len samples,
    with tapering defined by `window_name` and `window_width`.

    Parameters:
        window_name (str): Name of the window function (e.g., 'hanning', 'hamming', 'blackman')
        cf_len (int): Length of the correlation function (should be 2 * n_data - 1)
        window_width (int): Half-width of the active window region around zero-lag

    Returns:
        np.ndarray: A window array of shape (cf_len,) with zeros outside the taper.
    """
    zero_lag_idx = cf_len // 2  # Zero-lag index is in the center

    # Initialize full-length zero window
    window = np.zeros(cf_len)

    # Length of taper (must be odd)
    taper_len = 2 * window_width + 1

    # Generate desired window shape
    if window_name.lower() == 'hanning':
        taper = np.hanning(taper_len)
    elif window_name.lower() == 'hamming':
        taper = np.hamming(taper_len)
    elif window_name.lower() == 'blackman':
        taper = np.blackman(taper_len)
    else:
        raise ValueError(f"Unsupported window type: {window_name}")

    # Bounds of taper placement
    start = max(zero_lag_idx - window_width, 0)
    end = min(zero_lag_idx + window_width + 1, cf_len)

    # Slice taper appropriately if near edge
    taper_start = window_width - (zero_lag_idx - start)
    taper_end = taper_start + (end - start)

    window[start:end] = taper[taper_start:taper_end]

    return window

@extend2NDarray
def divide_and_swap(x,N):
    first = x[:N]
    second = x[N:]
    x[:len(second)] = second
    x[len(second):] = first
    return x

# @vectorize_along_axis
# def np2tuple(x)->List[Tuple]:
#     return tuple(x)

def merge_dataframes(dfs: List[pd.DataFrame], on: List[str], how: str = 'inner') -> pd.DataFrame:
    """
    Merge a list of dataframes on specified columns.
    
    Parameters:
    - dfs: List of pandas DataFrames to merge.
    - on: List of column names to merge on.
    - how: Type of join – 'inner', 'outer', 'left', 'right'. Default is 'inner'.

    Returns:
    - A single merged DataFrame.
    """
    if not dfs:
        raise ValueError("The list of dataframes is empty.")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=on, how=how)
    
    return merged

def group_floats_by_decimals(floats: np.ndarray, decimals: int) -> Dict[float, np.ndarray]:
    """
    Groups float numbers by rounding them to a fixed number of decimal places,
    but stores the original (unrounded) float values in the output.

    Parameters:
        floats (np.ndarray): Input array of float numbers.
        decimals (int): Number of decimal places to round to for grouping.

    Returns:
        Dict[float, np.ndarray]: {rounded_value: array of original float values}

    Example:
        >>> floats = np.array([0.512, 0.515, 0.518, 1.234, 1.236, 2.0001])
        >>> grouped = group_floats_by_decimals(floats, decimals=0)
        >>> for key, vals in grouped.items():
        ...     print(f"{key}: {vals}")
        1.0: [0.512 0.515 0.518 1.234 1.236]
        2.0: [2.0001]
    """
    from collections import defaultdict
    floats = np.asarray(floats)
    rounded_keys = np.round(floats, decimals)

    groups = defaultdict(list)
    for key, val in zip(rounded_keys, floats):
        groups[key].append(val)

    return {float(k): np.array(v) for k, v in groups.items()}

def gaussian_cluster_indices(floats: np.ndarray, n_components: int = 2, random_state: int = 0) -> Dict[float, np.ndarray]:
    """
    Cluster 1D float data into Gaussian components and return a mapping
    from cluster center to original indices.

    Parameters
    ----------
    floats : np.ndarray
        1D array of float values
    n_components : int, default=2
        Number of Gaussian clusters to fit
    random_state : int, default=0
        Random seed for reproducibility

    Returns
    -------
    Dict[float, np.ndarray]
        Keys = Gaussian centers (means)
        Values = np.ndarray of indices mapping to original floats
    """
    from sklearn.mixture import GaussianMixture
    floats = np.asarray(floats).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(floats)
    
    labels = gmm.predict(floats)
    centers = gmm.means_.flatten()
    
    # Sort centers for consistency
    sorted_idx = np.argsort(centers)
    centers = centers[sorted_idx]

    cluster_dict = {}
    for i, center_idx in enumerate(sorted_idx):
        indices = np.where(labels == center_idx)[0]
        cluster_dict[float(centers[i])] = indices

    return cluster_dict

def find_duplicates(arr:np.ndarray)->Dict[str,np.ndarray]:
    """
    Find duplicate integers in a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        1D array of integers.

    Returns
    -------
    dict
        {value: [positions]} for each duplicated value.
    """
    assert arr.ndim == 1, "Input array must be 1D."
    duplicates = {}

    # unique values, inverse indices, and counts
    unique_vals, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)

    # iterate over values that occur more than once
    for val, count in zip(unique_vals, counts):
        if count > 1:
            positions = np.where(arr == val)[0]# .tolist()
            duplicates[val] = positions

    return duplicates