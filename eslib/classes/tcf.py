from dataclasses import dataclass, field
from gc import enable
from typing import Optional, Tuple, TypeVar

import numpy as np
from scipy.fftpack import dct
from scipy.optimize import curve_fit
from tqdm import tqdm

from eslib.classes.timing import timing
from eslib.tools import convert
from scipy import signal
from eslib.functional import extend2NDarray

T = TypeVar('T', bound='TimeCorrelation')

@dataclass
class TimeCorrelation:
    """
    Class to compute time correlation functions between two arrays.

    Attributes
    ----------
    A : np.ndarray
        First input array.
    B : np.ndarray
        Second input array.
    _ready : bool, default False
        Flag indicating whether the correlation function is computed and ready to use.
    _tcf : np.ndarray, default None
        Computed time correlation function.

    Methods
    -------
    tcf(axis: Optional[int] = 0) -> np.ndarray:
        Computes and returns the time correlation function along the specified axis.

    Example
    -------
    >>> import numpy as np
    >>> A = np.random.random((100, 1))
    >>> B = np.random.random((200, 1))
    >>> tc = TimeCorrelation(A=A, B=B)
    >>> tcf_result = tc.tcf(axis=0)
    """

    A: np.ndarray
    B: np.ndarray
    # _ready: bool = field(default=False, init=False)
    # _tcf: np.ndarray = field(default=None, init=False)
    # _norm: np.ndarray = field(default=None, init=False)

    def __post_init__(self: T):
        """
        Initializes the TimeCorrelation instance and validates input array shapes.

        Raises
        ------
        AssertionError
            If input array shapes are not compatible.
        """
        Ashape = list(self.A.shape)[1:]
        Bshape = list(self.B.shape)[1:]

        assert self.A.ndim == self.B.ndim
        assert len(Ashape) == len(Bshape)  # redundant
        assert all([a == b for a, b in zip(Ashape, Bshape)])

    def tcf(self, axis: Optional[int] = 0) -> np.ndarray:
        """
        Computes the time cross-correlation function along the specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the correlation function. Default is 0.

        Returns
        -------
        np.ndarray
            Computed time correlation function.
        """
        # if self._ready:
        #     return self._tcf
        # else:
        #     self._tcf = correlate(self.A, self.B, axis=axis)
        #     self._ready = True
        #     return self._tcf.copy()
        return correlate(self.A, self.B, axis=axis)

@dataclass
class TimeAutoCorrelation:
    """
    Class to compute auto-correlation functions for a single array.

    Attributes
    ----------
    A : np.ndarray
        Input array for auto-correlation.
    _ready : bool, default False
        Flag indicating whether the auto-correlation function is computed and ready to use.
    _tcf : np.ndarray, default None
        Computed auto-correlation function.

    Methods
    -------
    tcf(axis: Optional[int] = 0) -> np.ndarray:
        Computes and returns the auto-correlation function along the specified axis.

    Example
    -------
    >>> import numpy as np
    >>> A = np.random.random((100, 1))
    >>> tac = TimeAutoCorrelation(A=A)
    >>> tcf_result = tac.tcf(axis=0)
    """
    
    A: np.ndarray

    # def __init__(self: T, A: np.ndarray):
    #     super().__init__(A=A, B=A.copy())

    def tcf(self: T, axis: Optional[int] = 0, mode: str = "half",normalize:str="none",axis_sum:Optional[int]=-1) -> np.ndarray:
        """
        Computes the time correlation function along the specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the correlation function. Default is 0.
        mode : str, optional
            Mode of the correlation function, either 'half' or 'full'. Default is 'half'.

        Returns
        -------
        np.ndarray
            Computed time correlation function.
        """
        if mode not in ["half","full"]:
            raise ValueError("`mode` can be only `half` or `full`")
        
        arr = my_autocorrelate(self.A, axis=axis) 
    
        if normalize == "component":
            norm = np.mean(self.A ** 2, axis=axis,keepdims=True) # normalized to 1 each component
            arr /= norm 
            test = np.take(arr,0,axis)
            assert np.allclose(test,1), "The auto-correlation function is not normalized to 1"
            # self._norm = norm
        elif normalize == "all":
            norm = np.sum(np.mean(self.A ** 2, axis=axis),axis=axis_sum) # normalized to 1 the sum fo all the components
            norm = np.expand_dims(norm, axis=(axis,axis_sum))
            arr /= norm
            test = np.sum(np.take(arr,0,axis),axis=axis_sum-1 if axis_sum>axis else axis_sum)
            assert np.allclose(test,1), "The auto-correlation function is not normalized to 1"
            # self._norm = norm
        else:
            pass
        if mode == "half":
            N = int(np.round(arr.shape[axis] / 2))  # the second half is noisy
            # arr = arr[:N]      
            arr = np.take(arr, np.arange(N), axis=axis)      
        return arr

# import cProfile
# import functools
# import numpy as np
# from typing import Optional

# # Decorator to profile a function and save the output to a file
# def profile_function(output_file: str):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             profiler = cProfile.Profile()
#             profiler.enable()  # Start profiling

#             result = func(*args, **kwargs)  # Execute the original function

#             profiler.disable()  # Stop profiling
#             profiler.dump_stats(output_file)  # Save the profiling data
#             print(f"Profile saved to {output_file}")
            
#             return result

#         return wrapper
#     return decorator

# # Use the decorator to profile the correlate function
# # @profile_function(output_file='correlate.profile.txt')
# def correlate(
#     A: np.ndarray, 
#     B: np.ndarray, 
#     axis: Optional[int] = 0) -> np.ndarray:
#     """
#     Computes the cross-correlation of two arrays along the specified axis.

#     Parameters
#     ----------
#     A : np.ndarray
#         First input array.
#     B : np.ndarray
#         Second input array.
#     axis : int, optional
#         Axis along which to compute the correlation. Default is 0.

#     Returns
#     -------
#     np.ndarray
#         Cross-correlation of the two input arrays.
#     """

#     # assert axis == 0, "not debugged yet with `axis` != 0"
#     assert A.ndim == B.ndim
#     shape_a = A.shape
#     len_a = shape_a[axis]
#     shape_b = B.shape
#     len_b = shape_b[axis]
    
#     if len_a >= len_b:
#         # Example (1):
#         # A.shape = (300,1)
#         # B.shape = (120,1)
#         a_wkspace = slice_along_axis(A, axis, end=len_b)
#         len_fft = 2*len_b # 240
#         len_tcf = len_b # 120
#         norm_tcf = np.arange(len_b, 0, -1, dtype=int) # [120,120,...,119,118,...3,2,1]
#     else:
#         # Example (2):
#         # A.shape = (100,1)
#         # B.shape = (240,1)
#         len_tcf = len_b         # 240
#         len_fft = len_a + len_b # 340
#         a_wkspace = A
#         norm_tcf = np.arange(len_b, 0, -1, dtype=int)          
#         norm_tcf = np.where(norm_tcf > len_a, len_a, norm_tcf) # [100,100,...,99,98,...3,2,1]

#     dims_to_append = np.arange(B.ndim-1, -1, -1, dtype=int)[axis]
#     norm_tcf = append_dims(norm_tcf, dims_to_append)   # (120,1) for Example (1), (171,1) for Example (2)
#     ftA = np.fft.rfft(a_wkspace, axis=axis, n=len_fft) # (121,1) for Example (1), (171,1) for Example (2)
#     ftB = np.fft.rfft(B, axis=axis, n=len_fft)         # (121,1) for Example (1), (171,1) for Example (2)
#     np.conj(ftA, out=ftA)
#     ftB *= ftA
#     del ftA # no longer needed
#     ftB = np.fft.irfft(ftB, axis=axis, n=len_fft) # (240,1) for Example (1), (340,1) for Example (2)
#     indices = np.arange(len_tcf)
#     ftB = np.take(ftB, indices, axis=axis) # overwrite ftB
#     return ftB / norm_tcf # (120,1) for Example (1), (240,1) for Example (2)

# @extend2NDarray
# def autocorrelate(data,mean=0.0,method='auto', normalize=False):
#     return correlate(data1=data,data2=data,mean1=mean,mean2=mean,method=method,normalize=normalize)

@extend2NDarray
def autocorrelate(data,*argv,**kwargs):
    return correlate(data1=data,data2=data,*argv,**kwargs)

# @extend2NDarray
# def autocorrelate(data,*argv,**kwargs):
#     return PL_correlate(data)

def PL_correlate(observable):
    nsteps = observable.shape[0]
    ft = np.fft.rfft(observable, axis=0)
    ft_ac = ft*np.conjugate(ft)
    autoCorr = np.fft.irfft(ft_ac,axis=0)[:int(int(nsteps/2)+1)]/np.average(observable**2,axis=0,keepdims=True)/nsteps
    # autoCorr = autoCorr.mean(axis=1)
    return autoCorr


def correlate(data1, data2, mean1=0.0, mean2=0.0, method='auto', normalize=False,mode="full"):
    """Correlation function of `data1` and `data2`.

    The mean of the input data can be shifted before calculating the CF using
    `mean1` and `mean2`. A provided value (useful if you know the true mean)
    will shift by the requested amount, while 'mean' will shift by the mean of
    the data itself.

    This uses `scipy.signal.correlate` internally:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
    and the ``method` flag is passed on directly to it.

    Arguments:
        data1, data2: 1D arrays to correlate, lengths n1 and n2
        mean1, mean2: a number or 'mean'
        method: string, passed directly to `scipy.signal.correlate`
        normalize: boolean, if True, normalize the CF to 1.0 at zero delay

    Returns:
        array, 1D, length n1 + n2 -1
    """
    
    if mode not in ["half","full","shift"]:
        raise ValueError("`mode` can be only `half`, `full` or `shift`")

    if (data1.ndim != 1) or (data2.ndim != 1):
        raise ValueError('1D arrays required.')

    # lengths of input data
    n1 = len(data1)
    n2 = len(data2)
    if n1 != n2:
        raise ValueError(f'Data must have the same length. Got {n1:d} and {n2:d}.')
    n_data = n1

    # additional normalization factor
    norm = n_data - np.abs(np.arange(1-n_data, n_data), dtype=float)

    # optionally shift the data by the mean - provided or determined
    if mean1 == 'mean':
        mean1 = data1.mean()
    if mean2 == 'mean':
        mean2 = data2.mean()
    if mean1 != 0.0:
        data1 = data1 - float(mean1)
    if mean2 != 0.0:
        data2 = data2 - float(mean2)

    # call SciPy's `correlate` and correct for the difference in the definition
    # of what we want vs what they have:
    # - invert the time axis, because they (NumPy as well) have the opposite sign of the lag
    # - divide by a normalization factor such that we actually have an average at each delay
    cf = signal.correlate(data1, data2, mode='full', method=method)[::-1] / norm

    assert len(cf) == 2*n_data - 1

    # normalize to 1.0 at zero delay if requested
    if normalize:
        cf /= cf[n_data-1]
    
    # assert np.argmax(cf) == n_data-1, "weird error"
        
    if mode == "half":
        # return only the second half of the correlation function
        cf = np.take(cf, np.arange(n_data-1,len(cf)))
        # assert np.argmax(cf) == 0, "weird error"
    elif mode == "shift":
        raise NotImplementedError("`mode` = `shift` not implemented yet")
    elif mode == "full":
        pass
    else:
        raise ValueError("`mode` can be only `half`, `full` or `shift`")
    
    lags = signal.correlation_lags(len(data1), len(data2), mode="full")

    return cf

def serial_fft(A: np.ndarray, axis:int=-1, len_fft:Optional[int]=None,enable:bool=False):
    
    # Determine the shape of A
    shape = A.shape
    
    # Create an empty array to store the results
    # Adjust the shape for the rfft result along the FFT axis
    result_shape = list(shape)
    if len_fft is None:
        len_fft = shape[axis]
    result_shape[axis] = len_fft // 2 + 1  # rfft output size for real input
    
    result = np.empty(result_shape, dtype=np.complex128)
    
    # Calculate the total number of iterations
    total_slices = np.prod(shape[:axis]) * np.prod(shape[axis+1:])
    
    # Iterate over all indices except the one along the FFT axis
    for idx in tqdm(np.ndindex(*shape[:axis], *shape[axis+1:]), total=total_slices, desc="Computing FFT",enable=enable):
        # Build a slice object to extract a sub-array along the FFT axis
        slc = list(idx[:axis]) + [slice(None)] + list(idx[axis:])
        slc = tuple(slc)
        
        # Perform FFT on the sliced array
        result[slc] = np.fft.rfft(A[slc], n=len_fft)
    
    return result

# @profile_function(output_file='autocorrelate.profile.txt')
def my_autocorrelate(A: np.ndarray, axis: Optional[int] = 0,loop:Optional[bool]=False) -> np.ndarray:
    """
    Computes the auto-correlation of an array along the specified axis
    when A == B.

    Parameters
    ----------
    A : np.ndarray
        Input array (since A == B, only one array is needed).
    axis : int, optional
        Axis along which to compute the correlation. Default is 0.

    Returns
    -------
    np.ndarray
        Auto-correlation of the input array.
    """

    shape_a = A.shape
    len_a = shape_a[axis]
    
    # Use the entire array A for FFT since A == B
    len_fft = 2 * len_a
    len_tcf = len_a
    norm_tcf = np.arange(len_a, 0, -1, dtype=int)
    
    dims_to_append = np.arange(A.ndim - 1, -1, -1, dtype=int)[axis]
    norm_tcf = append_dims(norm_tcf, dims_to_append)
    
    if loop:
        ftA = serial_fft(A, axis=axis, len_fft=len_fft,enable=True)
    else:
        ftA = np.fft.rfft(A, axis=axis, n=len_fft)
        

    # # Compute the FFT of A
    # if A.nbytes > 1e8 or loop:
    #     # Create an output array to accumulate the results
    #     ftA = np.zeros_like(A, shape=(len_tcf,) + A.shape[1:])
    #     # Iterate over all slices except along the axis we are performing the FFT
    #     it = np.nditer(np.zeros(A.shape[:axis] + A.shape[axis+1:]), flags=['multi_index'])
    #     while not it.finished:
    #         idx = list(it.multi_index)
    #         idx.insert(axis, slice(None))

    #         # Compute the FFT for the slice along the axis of interest
    #         ftA[tuple(idx)] = np.fft.rfft(A[tuple(idx)], axis=axis, n=len_fft)
            
    #         # Multiply by its conjugate (auto-correlation in Fourier space)
    #         ftA[tuple(idx)] *= np.conj(ftA[tuple(idx)])
            
    #         # Inverse FFT to get the correlation
    #         ftA[tuple(idx)] = np.fft.irfft(ftA[tuple(idx)], axis=axis, n=len_fft)
            
    #         # Slice the result to the length of the original signal and normalize
    #         ftA[tuple(idx)] = np.take(ftA[tuple(idx)], np.arange(len_tcf), axis=axis) / norm_tcf
            
    #         it.iternext()
    #     # ftA = result.copy()
    #     # del result
    # else:
    #     
    
    # ftA = np.fft.rfft(A, axis=axis, n=len_fft)

    # Multiply the FFT with its conjugate to get the power spectrum
    ftA *= np.conj(ftA)
    
    # Compute the inverse FFT to get the auto-correlation
    ftA = np.fft.irfft(ftA, axis=axis, n=len_fft)
    
    # Slice the result to get the correct length of the auto-correlation
    ftA = np.take(ftA, np.arange(len_tcf), axis=axis)
    
    return ftA / norm_tcf  # Normalize by the decreasing window size

def append_dims(arr, dims_to_append):
    for dim in dims_to_append:
        arr = np.expand_dims(arr, axis=dim)
    return arr


def append_dims(arr, ndims=1):
    """Return a view of the input array with `ndims` axes of
    size one appended.
    """
    return arr[(Ellipsis,) + ndims*(None,)] 

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

def dummy_correlation(A:np.ndarray,B:np.ndarray,std:Optional[bool]=False)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Compute the cross-correlation of two arrays using a simple implementation.
    
    Parameters
    ----------
    A : np.ndarray
        First input array.
    B : np.ndarray
        Second input array.
    std : bool, optional
        Flag indicating whether to compute the standard deviation as well, defaults to False.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the cross-correlation function, standard deviation (if std=True), and the count of elements in each correlation bin.
    """
    # Initialize arrays to store results
    tcf = np.full(B.shape,np.nan)
    if std:
        std_tcf = np.full(B.shape,np.nan)
    N = np.full(B.shape,np.nan)
    
    lenA = len(A)
    lenB = len(B)
    # Iterate over each element in B to compute cross-correlation
    for n in range(len(B)):
        try:
            start  = n
            end = min(lenA+n,lenB) 
            if lenA+n > lenB:
                pass
            # Compute the cross-correlation for the current bin
            arr = (A[:(end-start)] * B[start:end])
            tcf[n] = arr.mean()  # Mean of the cross-correlation
            if std:
                std_tcf[n] = arr.std()  # Standard deviation of the cross-correlation
            N[n] = len(arr)  # Count of elements in the current bin
        except:
            break
    
    assert not np.any(np.isnan(tcf)), "The correlation function contains NaN values"
    # Remove NaN values from the arrays
    # tcf = np.take(tcf,indices=~np.isnan(tcf).flatten(),axis=0)
    if std:
        std_tcf = std_tcf[ ~np.isnan(std_tcf) ]
    N = N[ ~np.isnan(N) ]

    # Return results based on whether standard deviation is computed
    if std:
        return tcf,std_tcf,N
    else:
        return tcf,N

def get_freq(dt: float, N: int, input_units: str = "femtosecond", output_units: str = "thz") -> np.ndarray:
    """
    Compute the frequency array from the timestep.

    Parameters:
    -----------
    dt: float
        Timestep in femtoseconds.
    N: int
        Length of the spectrum array.
    input_units: str, optional
        Units of the timestep. Default is "femtoseconds".
    output_units: str, optional
        Units of the frequency array. Default is "thz".

    Returns:
    --------
    freq: np.ndarray
        Frequency array.
    """
    # Convert timestep to seconds
    dt = convert(dt, "time", input_units, "second")

    # Compute the sampling rate in Hz
    sampling_rate = 1 / dt

    # Convert sampling rate to the desired units
    sampling_rate = convert(sampling_rate, "frequency", "hz", output_units)

    # Compute the frequency array
    freq = np.linspace(0, sampling_rate / 2, N)

    return freq

def compute_spectrum(
    autocorr: np.ndarray,
    axis: int = 1,
    pad: Optional[int] = 0,
    # method: Optional[str] = "dct",
    dt: Optional[float] = 1.0,
    shift: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectrum from the autocorrelation function.

    Parameters:
        autocorr (np.ndarray): Autocorrelation or correlation function (from `signal.correlate` with mode='full').
        axis (int): Axis along which to compute the spectrum.
        pad (int): Number of times to pad the autocorr size along the axis.
        method (str): 'rfft' (default) or 'dct'.
        dt (float): Sample spacing in time units.

    Returns:
        spectrum (np.ndarray): Frequency-domain representation.
        freq (np.ndarray): Positive frequency values (in Hz if dt is in seconds).
    """
    
    # Pad autocorrelation
    pad_width = [ (0, 0) ] * autocorr.ndim
    pad_size = pad * autocorr.shape[axis]
    pad_width[axis] = (pad_size, pad_size)
    autocorr = np.pad(autocorr, pad_width, mode='constant',constant_values=(0,0))
    
    if shift:
        func = extend2NDarray(np.fft.ifftshift)
        autocorr = func(autocorr,axis=axis)    
    
    
    # if method == "dct":
    #     from scipy.fft import dct, rfftfreq
    #     spectrum = dct(autocorr_padded, type=1, axis=axis)
    #     n = autocorr_padded.shape[axis]
    #     freq = np.arange(n) / (2.0 * (n - 1) * dt)
    # elif method == "rfft":
    from scipy.fft import rfft, rfftfreq
    spectrum = rfft(autocorr, axis=axis)
    freq = rfftfreq(autocorr.shape[axis], dt)
    # else:
    #     raise ValueError(f"Unsupported method: {method}")

    assert freq.shape[0] == spectrum.shape[axis], "Frequency axis length mismatch"
    return spectrum, freq


def compute_cyclic_derivative(arr: np.ndarray, axis: Optional[int]=0) -> np.ndarray:
    """
    Compute the derivative of a 1D array with periodic boundary conditions.

    Parameters:
        arr (numpy.ndarray): Input 1D array of function values.
        axis (int): Axis along which to take the derivative.

    Returns:
        numpy.ndarray: The derivative of the array.

    Notes:
        The derivative is computed using the formula:
            d/dx f(x) = [f(x+dx) - f(x-dx)] / 2*dx
        with periodic boundary conditions, i.e. f(x+L) = f(x) where L is the length of the array.
    """
    # Ensure arr is a numpy array
    arr = np.asarray(arr)
    # Compute the derivative with periodic boundary conditions
    return np.roll(arr, shift=-1, axis=axis) - arr

def compute_corr_time(arr: np.ndarray) -> np.ndarray:
    
    assert arr.ndim == 1, "Input array must be 1D"
    
    def func(time,A,tau,omega,phi):
        return A*np.exp(-time/tau)*np.sin(omega*time+phi)
    
    # def func(time,tau):
    #     return np.exp(-time/tau)# *np.sin(omega*time+phi)
    
    time = np.arange(len(arr))
    lb = [0, 0, 0, -np.pi]                  #  lower bounds
    ub = [np.inf, len(arr), np.inf, +np.pi] # higher bounds
    popt, pcov = curve_fit(func,time,arr,p0=[arr[0],10,10,0],bounds=(lb,ub))
    return popt[1], np.sqrt(pcov[1,1])
    
    lb = [0]                  #  lower bounds
    ub = [np.inf] # higher bounds
    popt, pcov = curve_fit(func,time,arr,p0=[10],bounds=(lb,ub))
    return popt[0], np.sqrt(pcov[0,0])
        
    
def main():
    """
    Main function to demonstrate the computation of time correlation functions.

    This function generates random arrays `A` and `B` of different lengths and computes the time correlation function using two different methods:
    1. Using the `TimeCorrelation` class with the FFT trick.
    2. Using a simple correlation function (`dummy_correlation`).

    The function measures the execution time of each method using the `timing` context manager and compares the results obtained from both methods.

    Raises
    ------
    AssertionError
        If the results obtained from both methods are not approximately equal.
    """
    dim = 1
    
    #########################
    # using FFT trick
    with timing():
        A = np.random.random((100,dim))
        corr = TimeCorrelation(A=A,B=A).tcf()
        autocorr = TimeAutoCorrelation(A=A).tcf()
    assert np.allclose(corr,autocorr), "Ostregeta, son' mica uguali qui!"

    #########################
    # using FFT trick
    with timing():
        A = np.random.random((100,dim))
        B = np.random.random((240,dim))
        test = TimeCorrelation(A=A,B=B)
        tcf_AB = test.tcf() # tcf_AB.shape: (240,1)
    with timing():
        tcf, N = dummy_correlation(A,B,std=False)
    with timing():
        tcf,std_tcf, N = dummy_correlation(A,B,std=True)
        # tcf = np.atleast_2d(tcf)
    assert tcf_AB.shape == tcf.shape, "Maremma zucchina, le 'shapes' sono diverse!"
    assert np.allclose(tcf_AB,tcf.reshape(-1,1)), "Ostregeta, son' mica uguali qui!"

    #########################
    # using FFT trick
    with timing():
        A = np.random.random((300,dim))
        B = np.random.random((120,dim))
        test = TimeCorrelation(A=A,B=B)
        tcf_BA = test.tcf() # tcf_AB.shape: (100,1)
    
    with timing():
        tcf, N = dummy_correlation(A,B,std=False)
    with timing():
        tcf,std_tcf, N = dummy_correlation(A,B,std=True)
    assert tcf_BA.shape == tcf.shape, "Oh 'de! Le 'shapes' sono diverse anche qui!"
    assert np.allclose(tcf_BA,tcf.reshape(-1,1)), "Eh la vacca, qui non va mica meglio!"

    #########################
    # Fourier Transform or the derivative (in the discrete case)
    # https://math.stackexchange.com/questions/1657756/is-there-in-discrete-fourier-transform-a-theorem-that-corresponds-to-transforms
    N      = 3000
    w      = 10
    x      = np.arange(N) 
    A      = np.sin(w*x**2)# +0.1*np.random.random(N)          # signal
    A     -= A.mean()                     # useful only for the autocorrelation functions
    fft    = np.fft.rfft(A)               # Fourier Transform of the signal
    derA   = np.gradient(A) # compute_cyclic_derivative(A) # Derivative of the signal
    fftder = np.fft.rfft(derA)            # Fourier Transform of the derivative of the signal
    freq   = np.fft.rfftfreq(N)           # Frequencies
    phases = np.exp(1.j*2*np.pi*freq)-1   # Phases
    test   = phases*fft                   # Hypothesis: `phases*fft`` should be equal to `fftder`

    import matplotlib.pyplot as plt

    # Plot of the Fourier Transform of the signal and of its derivative
    plt.plot(freq,np.abs(fft)   ,color="red"  ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$F\left[f\right]\left(\nu\right)$')
    plt.plot(freq,np.abs(fftder),color="blue" ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$F\left[\dot{f}\right]\left(\nu\right)$')
    plt.xlim(min(freq),max(freq))
    # plt.ylim(0,70)
    plt.xlabel("Frequency $\\nu$")
    plt.legend(facecolor='white', framealpha=1,edgecolor="black")
    plt.grid()
    plt.show()
    plt.cla() # clear the axes

    # Plot to show the relation between the Fourier Transform of the signal and the Fourier Transform of the derivative of the signal
    plt.plot(freq,np.abs(test)  ,color="red"  ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$\left(e^{i2\pi\nu} - 1\right) \cdot F\left[f\right]\left(\nu\right)$')
    plt.plot(freq,np.abs(fftder),color="blue" ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$F\left[\dot{f}\right]\left(\nu\right)$')
    plt.plot(freq,np.abs(test-fftder),color="black",linestyle="solid",linewidth=1,label="difference")
    plt.xlim(min(freq),max(freq))
    plt.xlabel("Frequency $\\nu$")
    plt.legend(facecolor='white', framealpha=1,edgecolor="black")
    plt.grid()
    plt.show()
    plt.cla() # clear the axes

    # Plot to show the continous limit
    tmp = 1.j*2*np.pi*freq
    plt.plot(freq,tmp.real  ,color="red"   ,linestyle="solid",linewidth=1,label=r'${\rm Re} \left(i2\pi\nu\right)$')
    plt.plot(freq,tmp.imag  ,color="blue"  ,linestyle="solid",linewidth=1,label=r'${\rm Im} \left(i2\pi\nu\right)$')
    plt.plot(freq,phases.real,color="red"  ,linestyle="dashed",linewidth=1,label=r'${\rm Re} \left(e^{i2\pi\nu} - 1\right)$')
    plt.plot(freq,phases.imag,color="blue" ,linestyle="dashed",linewidth=1,label=r'${\rm Im} \left(e^{i2\pi\nu} - 1\right)$')
    plt.xlim(min(freq),max(freq))
    plt.xlabel("Frequency $\\nu$")
    plt.legend(facecolor='white', framealpha=1,edgecolor="black")
    plt.grid()
    plt.show()
    plt.cla() # clear the axes

    # Auto-correlation function
    tmp      = fft*np.conjugate(fft)         
    tcf      = np.fft.irfft(tmp)           # Auto-correlation function of the signal
    tmp      = fftder*np.conjugate(fftder)
    tcfder   = np.fft.irfft(tmp)           # Auto-correlation function of the derivative of the signal

    # tcf /= tcf.max()
    # tcfder /= tcfder.max()
    
    assert np.allclose(tcf.imag,0), "Auto-correlation function is not real"
    assert np.allclose(tcfder.imag,0), "Auto-correlation function is not real"
    # I need to check that it's even too

    plt.plot(tcf   ,color="red"  ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$C\left[f\right]\left(t\right)$')
    plt.plot(tcfder,color="blue" ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$C\left[\dot{f}\right]\left(t\right)$')
    # plt.xlim(min(freq),max(freq))
    # plt.ylim(0,70)
    # plt.xscale("log")
    plt.xlabel("Time")
    plt.legend(facecolor='white', framealpha=1,edgecolor="black")
    plt.grid()
    plt.show()
    plt.cla() # clear the axes

    spectrum     = np.fft.rfft(tcf)
    spectrum_der = np.fft.rfft(tcfder)

    assert np.allclose(spectrum.imag,0), "Spectrum is not real"
    assert np.allclose(spectrum_der.imag,0), "Spectrum is not real"

    # plt.plot(freq,spectrum    ,color="red"  ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$S\left[f\right]\left(\omega\right)$')
    # plt.plot(freq,spectrum_der,color="blue" ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$S\left[\dot{f}\right]\left(\omega\right)$')
    # # plt.xlim(min(freq),max(freq))
    # # plt.ylim(0,70)
    # # plt.xscale("log")
    # plt.xlabel("Frequency $\\nu$")
    # plt.legend(facecolor='white', framealpha=1,edgecolor="black")
    # plt.grid()
    # plt.show()
    # plt.cla() # clear the axes

    tmp = spectrum * phases*np.conjugate(phases)

    assert np.allclose(tmp,spectrum_der), "Spectra are not the same"
    assert np.allclose(tmp.imag,0), "Spectrum is not real"

    plt.plot(freq,tmp.real    ,color="red"  ,linestyle="solid",alpha=0.5,linewidth=1,label=r'${\rm Re} \left(e^{i2\pi\nu} - 1\right)^2 S\left[f\right]\left(\omega\right)$')
    plt.plot(freq,spectrum_der,color="blue" ,linestyle="solid",alpha=0.5,linewidth=1,label=r'$S\left[\dot{f}\right]\left(\omega\right)$')
    plt.plot(freq,np.abs(tmp-spectrum_der),color="black",linestyle="solid",linewidth=1,label="difference")
    plt.xlabel("Frequency $\\nu$")
    plt.legend(facecolor='white', framealpha=1,edgecolor="black")
    plt.grid()
    plt.show()
    plt.cla() # clear the axes


    #########################
    # spectrum
    method = "dct"
    
    x = np.arange(N)
    A = np.exp(-x)*np.cos(x)
    A =  np.random.random((1,N)) # A.reshape((1,N)) # +
    # A -= A.mean(axis=1)

    A =  np.random.random(N)
    fft = np.fft.rfft(A)
    freq = np.fft.rfftfreq(N)
    assert np.allclose(freq,np.arange(len(fft))/N)
    phase = np.exp(1.j*2*np.pi/N)
    discrete_phase = np.exp(1.j*2*np.pi*freq)-1
    plt.plot(discrete_phase*fft,color="red",label="f")
    derA = compute_cyclic_derivative(A,axis=0)
    plt.plot(np.fft.rfft(derA),color="blue",label="$\\nabla$f")
    plt.legend()
    plt.show()

    np.allclose(tmp,fftder)


    tacf = TimeAutoCorrelation(A.copy())
    tcf = tacf.tcf(axis=1)
    spectrum, freq = compute_spectrum(tcf,axis=1,method=method)

    grad = compute_cyclic_derivative(A.copy(),axis=1)
    tacf_grad = TimeAutoCorrelation(grad)
    tcf_grad = tacf_grad.tcf(axis=1)
    spectrum_grad, freq_grad = compute_spectrum(tcf_grad,axis=1,method=method)
    
    spectrum2 = (spectrum * (2*np.pi*freq)**2).flatten()
    spectrum_grad = spectrum_grad.flatten()

    discrete_freq = N*(np.exp(1.j*2*np.pi*freq/N)-1)



    assert np.allclose(freq,freq_grad), "Frequencies are not the same"
    # diff = spectrum * freq**2-spectrum_grad
    # assert np.allclose(spectrum2,spectrum_grad), "Spectrum is not the same"

    
    # plt.plot(freq,np.absolute(spectrum2),color="blue",linestyle="--",label="f")
    plt.plot(freq,-2*np.pi*np.absolute(spectrum_grad),color="red",linestyle="dotted",label="$\\nabla$f")
    # plt.plot(freq,discrete_freq,color="green",label="$1-e^{-i\\omega}$")
    tmp = (spectrum * discrete_freq**2).flatten()
    plt.plot(freq,np.absolute(tmp),color="purple",label="good")
    # plt.plot(freq,np.absolute(tmp-spectrum_grad),color="black",label="diff")
    # plt.xscale("log")
    plt.legend()
    plt.show()

    return 

#---------------------------------------#
if __name__ == "__main__":
    main()


