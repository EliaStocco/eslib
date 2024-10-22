from cProfile import label
import numpy as np
from typing import TypeVar, Union
T = TypeVar('T',bound="ElectricField")

TimeType = Union[float, np.ndarray]
class ElectricField:

    amp   : np.ndarray
    freq  : float
    phase : float
    peak  : float
    sigma : float

    def __init__(self:T, amp=None, freq=None, phase=None, peak=None, sigma=None):
        self.amp = amp if amp is not None else np.zeros(3)
        self.freq = freq if freq is not None else 0.0
        self.phase = phase if phase is not None else 0.0
        self.peak = peak if peak is not None else 0.0
        self.sigma = sigma if sigma is not None else np.inf

    def Efield(self:T,time:TimeType):
        """Get the value of the external electric field (cartesian axes)"""
        if hasattr(time, "__len__"):
            return np.outer(self._get_Ecos(time) * self.Eenvelope(time), self.amp)
        else:
            return self._get_Ecos(time) * self.Eenvelope(time) * self.amp

    def _Eenvelope_is_on(self):
        return self.peak > 0.0 and self.sigma != np.inf

    def Eenvelope(self:T,time:TimeType):
        """Get the gaussian envelope function of the external electric field"""
        # https://en.wikipedia.org/wiki/Normal_distribution
        if self._Eenvelope_is_on():
            x = time  # indipendent variable
            u = self.peak  # mean value
            s = self.sigma  # standard deviation
            return np.exp(
                -0.5 * ((x - u) / s) ** 2
            )  # the returned maximum value is 1, when x = u
        else:
            return 1.0

    def _get_Ecos(self:T, time:TimeType):
        """Get the sinusoidal part of the external electric field"""
        # it's easier to define a function and compute this 'cos'
        # again everytime instead of define a 'depend_value'
        return np.cos(self.freq * time + self.phase)
    
    def derivative(self:T,time:TimeType):
        """Get the derivative of the external electric field"""
        return - np.outer( self.amp , (
            self.freq * np.sin(self.freq * time + self.phase) +
            (time-self.peak)/self.sigma**2 * np.cos(self.freq * time + self.phase)
        ) * self.Eenvelope(time) ).T

if __name__ == "__main__":
    # Example usage:
    
    efield = ElectricField(
        amp=np.array([1.0, 0.0, 0.0]),
        freq=1.0,
        phase=0.0,
        peak=50.0,
        sigma=10.0
    )

    Tmax = 100
    dt = 0.001
    time = np.linspace(0,Tmax,int(Tmax/dt))
    E = efield.Efield(time)[:,0]

    dEdt = np.gradient(E)/dt

    dEdt_exact = efield.derivative(time)[:,0]

    import matplotlib.pyplot as plt
    # plt.plot(time, E,label="E")
    plt.plot(time, dEdt,label="dE/dt (num)")
    plt.plot(time, dEdt_exact,label="dE/dt (exact)")
    plt.legend()
    plt.show()
    
    print(np.linalg.norm(dEdt - dEdt_exact))
    # np.allclose(dEdt,dEdt_exact), "Derivative is not correct"