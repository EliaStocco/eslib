import numpy as np

def reshape_into_blocks(data: np.ndarray, N: int) -> np.ndarray:
    """
    Reshape a numpy array into N blocks, discarding any excess elements.

    Parameters:
        data (numpy.ndarray): The input array to be reshaped.
        N (int): The number of blocks to reshape the data into.

    Returns:
        numpy.ndarray: The reshaped array containing N blocks.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> N = 3
        >>> reshaped_data = reshape_into_blocks(data, N)
        >>> print(reshaped_data)
        [[1 2]
         [3 4]
         [5 6]]
    """
    # Calculate the number of elements per block
    elements_per_block = len(data) // N
    
    # Truncate data to a length divisible by N
    truncated_length = elements_per_block * N
    truncated_data = data[:truncated_length]
    
    # Reshape the truncated data into N blocks
    reshaped_data = truncated_data.reshape((N, -1))
    
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
