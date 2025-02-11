import numpy as np
import matplotlib.pyplot as plt

# https://opg.optica.org/as/abstract.cfm?uri=as-50-8-1047

# List of files and corresponding factors
files = ['0-1000.csv', '1000-4000.csv', '4000-15000.piece=1.csv', '4000-15000.piece=2.csv', '4000-15000.piece=3.csv']
factors = [1, 1, 1, 100, 3000]
data = [None] * len(files)

# Read and process the data
for n, file in enumerate(files):
    data[n] = np.loadtxt(file, delimiter=",")
    data[n][:,1] /= factors[n]

# Stack all data into a single NumPy array
data = np.vstack(data)

data = data[np.argsort(data[:, 0])]

np.savetxt("water-IR.experimental.csv", data, fmt='%24.12f')


# Plot the data
fig, axs = plt.subplots(1, 2, figsize=(6,3))

# Linear plot
axs[0].plot(data[:, 0], data[:, 1],color="blue",linewidth=1)
axs[0].set_xlim(0,5000)
axs[0].set_ylim(0,None)
axs[0].set_xlabel('frequency [cm$^{-1}$]')
axs[0].set_ylabel('absorption [L/mol cm]')
axs[0].grid(True)

# Log-log plot
axs[1].plot(data[:, 0], data[:, 1], color="red",linewidth=1)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlim(1,1.5e4)
axs[1].set_ylim(2e-2,150)
axs[1].set_xlabel('frequency [cm$^{-1}$]')
# axs[1].set_ylabel('absorption [L/mol cm]')
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("water-IR.experimental.pdf",bbox_inches='tight')
plt.savefig("water-IR.experimental.png",dpi=300,bbox_inches='tight')