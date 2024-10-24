import numpy as np


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