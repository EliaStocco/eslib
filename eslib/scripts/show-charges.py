import numpy as np
from ase import Atoms
from ase.io import read
from icecream import ic

atoms:Atoms = read("test.extxyz",index=0)

symbols = atoms.get_chemical_symbols()
# ic(symbols)

charges = np.loadtxt("charges.txt")
# ic(charges.shape)

print("Nb: ",charges[:,0:6].mean()," pm ",charges[:,0:6].std())
print("Li: ",charges[:,6:12].mean()," pm ",charges[:,6:12].std())
print(" O: ",charges[:,12:].mean()," pm ",charges[:,12:].std())

print("done")