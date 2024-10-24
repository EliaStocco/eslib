from dataclasses import dataclass, field
from typing import Optional, TypeVar

import numpy as np
import pandas as pd

from eslib.classes.tcf import TimeAutoCorrelation
from eslib.tools import convert, element_wise_multiplication

T = TypeVar('T', bound='Spectrum')

@dataclass
class Spectrum(TimeAutoCorrelation):

    _spectrum_ready: bool = field(default=False, init=False)
    # _raw_spectrum: np.ndarray = field(default=None, init=False)
    _spectrum: np.ndarray = field(default=None, init=False)
    _spectrum_imag: np.ndarray = field(default=None, init=False)
    # _window: np.ndarray = field(default=None, init=False)

    def __init__(self: T, A: np.ndarray):
        super().__init__(A=A)

    def tcf(self: T, axis: Optional[int] = 0) -> np.ndarray:
        return super().tcf(axis=axis,mode="full")

    def spectrum(self:T,axis: Optional[int] = 0, autocorr: Optional[np.ndarray] = None)-> np.ndarray:
        """
        Computes the spectrum
        """
        if self._spectrum_ready:
            return self._spectrum
        else:
            if autocorr is None:
                autocorr = self.tcf(axis)
            
            # self._raw_spectrum = np.fft.rfft(autocorr,axis=axis).real
            
            # apply padding and windowing
            # if method.lower() != "none":
            #     func = getattr(np, method)
            #     window = np.zeros(autocorr.shape[axis])
            #     window[:M] = func(M,**kwargs)
            #     self._window = window

            #     autocorr = element_wise_multiplication(window,autocorr,axis)

            tmp = np.fft.rfft(autocorr,axis=axis)
            self._spectrum = tmp.real
            self._spectrum_imag = tmp.imag
            # else:
            #     self._spectrum = self._raw_spectrum.copy()
            
            self._spectrum_ready = True
        return self._spectrum.copy()
    
    # def to_json(self:T):
    #     return {
    #         "spectrum" : self._spectrum.copy(),
    #         # "raw-spectrum" : self._raw_spectrum.copy(),
    #         "window" : self._window.copy(),
    #     }




    
    # def dataframe(self,dt:float,spectrum:Optional[np.ndarray]=None,axis: Optional[int] = 0)->pd.DataFrame:
    #     """
    #     Returns a dataframe containing the frequencies (in THz) and the spectrum.
    #     The time-step `dt` is assumed to be in 'fs'.
    #     """
    #     df = pd.DataFrame(columns=['freq [THz]','spectrum','norm. spectrum'])
    #     if spectrum is None:
    #         spectrum = self.spectrum(axis=axis)
    #     freq = np.fft.rfftfreq(n=spectrum.shape[axis],d=dt) * 1000
    #     df['freq [THz]'] = freq
    #     df['spectrum'] = spectrum
    #     df['norm. spectrum'] = spectrum / np.linalg.norm(spectrum,axis=axis)
    #     return df