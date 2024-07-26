from cProfile import label
from dataclasses import dataclass, field
import numpy as np
from scipy.fftpack import dct
from eslib.tools import convert
import matplotlib.pyplot as plt
from typing import Optional, TypeVar, Tuple# , Callable, Any
# from scipy.signal import correlate as scipy_correlate
# import functools

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
        if self._ready:
            return self._tcf
        else:
            # if axis != 0:
            #     self.A = np.moveaxis(self.A, axis, 0)
            #     self.B = np.moveaxis(self.B, axis, 0)
            self._tcf = correlate(self.A, self.B, axis=axis)
            # if axis != 0:
            #     self.A    = np.moveaxis(self.A, 0, axis)
            #     self.B    = np.moveaxis(self.B, 0, axis)
            #     self._tcf = np.moveaxis(self._tcf, 0, axis)
            self._ready = True
            return self._tcf.copy()

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

    def __init__(self: T, A: np.ndarray):
        super().__init__(A=A, B=A.copy())

    def tcf(self: T, axis: Optional[int] = 0, mode: str = "half",normalize:bool=True) -> np.ndarray:
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
        arr:np.ndarray = super().tcf(axis)
        if normalize:
            arr = arr / np.mean(self.A ** 2, axis=axis,keepdims=True)  # normalized to 1
            assert np.allclose(np.take(arr,0,axis),1), "The auto-correlation function is not normalized to 1"
        if mode == "half":
            N = int(arr.shape[axis] / 2)  # the second half is noisy
            # arr = arr[:N]      
            arr = np.take(arr, np.arange(N), axis=axis)      
        return arr

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

    # assert axis == 0, "not debugged yet with `axis` != 0"
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
    indices = np.arange(len_tcf)
    tmp = np.take(tmp, indices, axis=axis)
    out = tmp / norm_tcf # (120,1) for Example (1), (240,1) for Example (2)
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
    axis: int,
    pad: Optional[int] = 0,
    method: Optional[str] = "dct",
    dt: Optional[float] = 1,
) -> np.ndarray:
    """
    Compute the spectrum from the autocorrelation function.

    Parameters:
        autocorr (np.ndarray): The autocorrelation function.
        axis (int, optional): The axis along which to compute the spectrum. Default is 1.
        pad (int, optional): The number of zeros to add before the autocorrelation function in units of `autocorr.shape[axis]`. Default is 0.

    Returns:
        np.ndarray: The spectrum.
    """
    # Add padding to the autocorrelation function
    N = autocorr.shape[axis]*pad
    padding = np.zeros(N)
    if axis == 1:
        autocorr = np.asarray([ np.append(a,padding) for a in autocorr ])
    else:
        raise ValueError("not implemented yet") 

    # Compute the discrete cosine transform of the autocorrelation function
    # along the specified axis
    
    if method == "dct":
        spectrum = dct(autocorr, type=1, axis=axis)
        n = autocorr.shape[axis]
        freq = np.arange(n) / (2.0 * (n - 1) * dt)
    elif method == "rfft":
        spectrum = np.fft.rfft(autocorr, axis=axis)
        freq = np.fft.rfftfreq(autocorr.shape[axis],dt)
    assert len(freq) == spectrum.shape[axis]
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


