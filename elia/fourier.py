import numpy as np
from numpy.fft import fft, rfft
from numpy.fft import fftfreq, rfftfreq
import matplotlib.pyplot as plt

class FourierAnalyzer:
    def __init__(self, time, signal, real=None):

        self.time = np.asarray(time)
        self.signal = np.asarray(signal)

        if real is None:
            real = np.all( signal == np.conjugate(signal))

        if real :
            self.functions = {
                "fft"  : lambda s: rfft(s,axis=0),
                "freq" : rfftfreq,
            }
        else :
            self.functions = {
                "fft"  : lambda s: fft(s,axis=0),
                "freq" : fftfreq,
            }


        self.compute_fourier_transform()

        # self.summary()

    def compute_fourier_transform(self):
        self.freq = self.functions['freq'](len(self.time), self.time[1] - self.time[0])
        self.omega = 2*np.pi*self.freq
        self.fft = self.functions['fft'](self.signal)
        self.spectrum = np.absolute(self.fft)
        return
    
    def summary(self):

        print("\n\tSummary:")
        attrs = ["time","signal","freq","omega","fft"]
        for k in attrs:
            obj = getattr(self,k)
            print("\t{:>20s}:".format(k),obj.shape)
        print("\n")

    def plot_time_series(self):
        plt.plot(self.time, self.signal)
        plt.title('Original Time Series')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_fourier_transform(self):
        plt.plot(self.freq, self.fft)
        plt.title('Fourier Transform')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_power_spectrum(self):
        freq = self.freq[self.freq >= 0]
        amp = 1.0/len(self.time) * np.absolute(self.fft)
        plt.plot(freq, amp)
        plt.title('Power Spectrum (Positive Frequencies Only)')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.show()

if __name__ == "__main__":
    # Example usage:
    np.random.seed(42)
    time = np.linspace(0, 10, 1000)
    signal = 2 * np.sin(2 * np.pi * 1 * time) + 1.5 * np.sin(2 * np.pi * 3 * time) + np.random.normal(scale=0.5, size=len(time))

    fourier_analyzer = FourierAnalyzer(time, signal)
    fourier_analyzer.plot_time_series()

    frequencies, fourier_transform = fourier_analyzer.compute_fourier_transform()
    fourier_analyzer.plot_fourier_transform()

    fourier_analyzer.plot_power_spectrum()
    
    print("\n\tJob done :)\n")
