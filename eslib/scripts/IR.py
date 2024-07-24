# Created by Alan M Lewis. A script to create IR spectra from a dipole time-series 

import numpy as np
from scipy.fft import rfft, rfftfreq,irfft
from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt 

# Number of MD trajectories to be averaged over
n_runs = 5
# Multiples of the correlation function length filled with zeros
n_pad = 5

# Read in data

# Assumes data stored with dimensions (nruns, length_of_runs, 3)
mu = np.load('dipole.n=0.npy')[None,:,:]

# Define variables
mu = mu - np.average(mu,axis=1)[:,None,:]

# Define time constants
t_corr = 5.0 # Length of Correlation Function (ps)
dt = 0.01 # Timestep (ps)
ncorr = int(t_corr / dt) + 1 # Length of CF (steps)
t0_max = len(mu[0,:,0]) - ncorr # Last possible t0
ts = np.linspace(0,t_corr,num=ncorr)
freq = rfftfreq(ncorr+(ncorr-1)*n_pad,dt)*33.35641

# Define Window
window = np.zeros(ncorr)
wl = int(8*ncorr/8)
window[:wl] = kaiser(2*wl,8)[wl:]/n_runs

# Define functions

# Build correlation function from time series
def corr_from_ft(data):
    ft = rfft(data,axis = 1)
    corr_ft = irfft(ft*np.conjugate(ft),axis=1)
    corr_ft = corr_ft[:,:ncorr,:]
    
    # Apply Window
    corr_ft *= window[None,:,None]
    
    return corr_ft

# Build spectrum from correlation function
def spectrum_from_corr(corr):

    # Add Padding

    pad = (1,ncorr-1,3)
    for i in range(n_pad):
        corr = np.concatenate((corr,np.zeros(pad)),axis=1)

    # Calculate Spectra
    spec = rfft(corr,axis=1).real
    norm = np.max(np.average(spec,axis=0),axis=0)
    eb = np.std(spec,axis=0)/np.sqrt(n_runs)

    return spec,norm,eb

# Isotropic Part

corr = corr_from_ft(mu)

np.savetxt('corr.out',np.column_stack([ts,np.average(corr,axis=0)]))

# norm allows us to normalise the spectra so that the maximum intensity is 1
# eb is the standard error in the mean at each point in the spectrum
spec,norm,eb = spectrum_from_corr(corr)

# Spectrum is output for x, y and z polarized light, in each case, with the mean followed by the errors in the mean at each frequency.
print(eb.shape,norm.shape,spec.shape)
np.savetxt('spectrum.out',np.column_stack([freq,np.average(spec[:,:,0],axis=0)/norm[0],eb[:,0]/norm[0],np.average(spec[:,:,1],axis=0)/norm[1],eb[:,1]/norm[1],np.average(spec[:,:,2],axis=0)/norm[2],eb[:,2]/norm[2]]))

spectrum = np.loadtxt('spectrum.out')

fig, ax = plt.subplots(1, figsize=(10, 4))

# ax.plot(freq,y,label="raw",color="red")
ax.plot(spectrum,color="blue", marker='.', markerfacecolor='blue', markersize=5)

ax.grid()
plt.tight_layout()
plt.show()

pass