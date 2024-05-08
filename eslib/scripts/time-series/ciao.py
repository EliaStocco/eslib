import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

data = np.load("potential.npy")[:10000]
data = data.reshape((10,-1)).T
# nsteps = 100000
# data = _data[:nsteps]
data -= data.mean()
ic(data.shape)
dt = 0.1
fft = np.fft.rfft(data,axis=0)
ic(fft.shape)
ft_ac = fft * np.conjugate(fft)
ic(ft_ac.shape)
autocorr = np.fft.irfft(ft_ac,axis=0)[:int(int(len(data)/2)+1)]/np.average(data**2,axis=0)/len(data)
ic(autocorr.shape)
x = np.arange(len(autocorr))

# plt.show()

old = autocorr.copy()
plt.plot(x,old.mean(axis=1),color="red")

data = np.load("potential.npy")[:10000]
# data = data.reshape((-1,10))
# nsteps = 100000
# data = _data[:nsteps]
data -= data.mean()
ic(data.shape)
dt = 0.1
fft = np.fft.rfft(data,axis=0)
ic(fft.shape)
ft_ac = fft * np.conjugate(fft)
ic(ft_ac.shape)
autocorr = np.fft.irfft(ft_ac,axis=0)[:int(int(len(data)/2)+1)]/np.average(data**2,axis=0)/len(data)
ic(autocorr.shape)
x = np.arange(len(autocorr))
plt.plot(x,autocorr,color="blue")
plt.show()

pass
