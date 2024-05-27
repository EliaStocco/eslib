from dataclasses import dataclass, field
import numpy as np
from typing import Optional, TypeVar, Tuple
from scipy.signal import correlate

# Try to import the timing function from the eslib.classes.timing module
try:
    from eslib.classes.timing import timing
# If ImportError occurs (module not found), define a dummy timing function
except ImportError:
    def timing(func):
        return func  # Dummy timing function that returns the input function unchanged


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
    _ready: bool = field(default=False, init=False)
    _tcf: np.ndarray = field(default=None, init=False)

    def __post_init__(self:T):
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
        assert len(Ashape) == len(Bshape) # redundant
        assert all( [ a==b for a,b in zip(Ashape,Bshape) ])

    @property
    def tcf(self:T,axis: Optional[int] = 0)->np.ndarray:
        """
        Computes the time correlation function along the specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the correlation function. Default is 0.

        Returns
        -------
        np.ndarray
            Computed time correlation function.
        """
        if self._ready:
            return self._tcf
        else:
            self._tcf = correlate(self.A, self.B, axis=axis)
            self._ready = True
            return self._tcf.copy()

@dataclass
class TimeAutoCorrelation(TimeCorrelation):
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

    def __post_init__(self:T):
        """
        Initializes the TimeAutoCorrelation instance and validates the input array.

        Raises
        ------
        AssertionError
            If the input array shape is not compatible.
        """
        self.B = self.A  # Ensure A and B are the same for auto-correlation
        super().__post_init__()


def correlate(
    A: np.ndarray, 
    B: np.ndarray, 
    axis: Optional[int] = 0) -> np.ndarray:
    """
    Computes the cross-correlation of two arrays along the specified axis.

    Parameters
    ----------
    A : np.ndarray
        First input array.
    B : np.ndarray
        Second input array.
    axis : int, optional
        Axis along which to compute the correlation. Default is 0.

    Returns
    -------
    np.ndarray
        Cross-correlation of the two input arrays.
    """

    assert A.ndim == B.ndim
    shape_a = A.shape
    len_a = shape_a[axis]
    shape_b = B.shape
    len_b = shape_b[axis]
    
    if len_a >= len_b:
        # Example (1):
        # A.shape = (300,1)
        # B.shape = (120,1)
        a_wkspace = slice_along_axis(A, axis, end=len_b)
        len_fft = 2*len_b # 240
        len_tcf = len_b # 120
        norm_tcf = np.arange(len_b, 0, -1, dtype=int) # [120,120,...,119,118,...3,2,1]
    else:
        # Example (2):
        # A.shape = (100,1)
        # B.shape = (240,1)
        len_tcf = len_b         # 240
        len_fft = len_a + len_b # 340
        a_wkspace = A
        norm_tcf = np.arange(len_b, 0, -1, dtype=int)          
        norm_tcf = np.where(norm_tcf > len_a, len_a, norm_tcf) # [100,100,...,99,98,...3,2,1]

    dims_to_append = np.arange(B.ndim-1, -1, -1, dtype=int)[axis]
    norm_tcf = append_dims(norm_tcf, dims_to_append)   # (120,1) for Example (1), (171,1) for Example (2)
    ftA = np.fft.rfft(a_wkspace, axis=axis, n=len_fft) # (121,1) for Example (1), (171,1) for Example (2)
    ftB = np.fft.rfft(B, axis=axis, n=len_fft)         # (121,1) for Example (1), (171,1) for Example (2)
    np.conj(ftA, out=ftA)
    ftB *= ftA
    tmp = np.fft.irfft(ftB, axis=axis, n=len_fft) # (240,1) for Example (1), (340,1) for Example (2)
    out = tmp[:len_tcf] / norm_tcf # (120,1) for Example (1), (240,1) for Example (2)
    return out

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
    
    # Remove NaN values from the arrays
    tcf = tcf [ ~np.isnan(tcf) ]
    if std:
        std_tcf = std_tcf[ ~np.isnan(std_tcf) ]
    N = N[ ~np.isnan(N) ]

    # Return results based on whether standard deviation is computed
    if std:
        return tcf,std_tcf,N
    else:
        return tcf,N


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
        B = np.random.random((240,dim))
        test = TimeCorrelation(A=A,B=B)
        tcf_AB = test.tcf # tcf_AB.shape: (240,1)
    with timing():
        tcf, N = dummy_correlation(A,B,std=False)
    with timing():
        tcf,std_tcf, N = dummy_correlation(A,B,std=True)
    assert tcf_AB.shape == tcf.shape, "Maremma zucchina, le 'shapes' sono diverse!"
    assert np.allclose(tcf_AB,tcf.reshape(-1,1)), "Ostregeta, son' mica uguali qui!"

    #########################
    # using FFT trick
    with timing():
        A = np.random.random((300,dim))
        B = np.random.random((120,dim))
        test = TimeCorrelation(A=A,B=B)
        tcf_BA = test.tcf # tcf_AB.shape: (100,1)
    
    with timing():
        tcf, N = dummy_correlation(A,B,std=False)
    with timing():
        tcf,std_tcf, N = dummy_correlation(A,B,std=True)
    assert tcf_BA.shape == tcf.shape, "Oh 'de! Le 'shapes' sono diverse anche qui!"
    assert np.allclose(tcf_BA,tcf.reshape(-1,1)), "Eh la vacca, qui non va mica meglio!"

#---------------------------------------#
if __name__ == "__main__":
    main()


