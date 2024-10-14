#Subtract avg values s.t. correlation function will go to zero.
from re import A
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt 

from eslib.classes.atomic_structures import AtomicStructures

# vel = AtomicStructures.from_file(file="vel.pickle")
# observable:np.ndarray = vel.get("positions")

# traj = AtomicStructures.from_file(file="nve.n=0.pickle")
# observable = traj.get("potential")[:-1]

observable = np.load("dipole.n=0.npy")

# observable = observable[50000:-1,0]

observable -= np.mean(observable,axis=0,keepdims=True)
observable = np.gradient(observable,axis=0)/0.1

nsteps = observable.shape[0]
cutoff = 0.1
npad = nsteps
dt = 0.1 #fs

# Compute autocorrelation function for the observable with FT trick.
ft = np.fft.rfft(observable, axis=0)
ft_ac = ft*np.conjugate(ft)
autoCorr = np.fft.irfft(ft_ac,axis=0)[:int(int(nsteps/2)+1)]/np.average(observable**2,axis=0,keepdims=True)/nsteps
autoCorr = autoCorr.mean(axis=1)
# autoCorr = autoCorr[:,0]


#Define window to be used with DCT below.
wind = np.hanning(int(nsteps*cutoff))
wind = wind[int(len(wind)/2):]
wind = np.append(wind, np.zeros(int(nsteps/2)+1 - len(wind)))

# plt.plot(autoCorr)
# plt.plot(autoCorr*wind)
# plt.plot(wind)
# plt.xlim(0,None)
# plt.show()

#Get the spectrum from DCT of windeowed autocorr (real part)
corrFT = dct(np.append(autoCorr*wind, np.zeros(npad)), type=1)
spectrum = corrFT.real

#Frequencies for DCT.
freq = np.fft.rfftfreq(2*int(int(nsteps/2)+npad)-1,dt)*1000*33.35641 #cm-1

x = freq
y = spectrum[:-1]#  *(np.square(freq))
y /= np.max(y)
plt.plot(x[1:],y[1:])
plt.xlim(-100,4500)
plt.ylim(0,1)
plt.show()

pass