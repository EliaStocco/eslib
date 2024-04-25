from ase.io import read
from ase import Atoms
import numpy as np
from eslib.classes.normal_modes import NormalModes
from eslib.geometry import angle_between_vectors
from icecream import ic
from ase.geometry import distance
import pandas as pd

stdau:Atoms = read("../start.au.extxyz")
std:Atoms = read("../start.ang.extxyz")
inv:Atoms = read("relax/start.ang.extxyz")

direction = np.asarray( inv.get_positions() - std.get_positions() ).flatten()
dist = np.linalg.norm(direction)
direction /= dist
# ic(distance)

# delta = std.get_positions() - inv.get_positions()
# dist = np.sqrt(np.square(delta)).sum(axis=1).sum()
# ic(distA," ang")
ic(dist," ang")

test = np.asarray( inv.get_positions().flatten() - ( std.get_positions().flatten() + direction * dist) )
ic(test)

nm = NormalModes.from_pickle("../eslib.pickle")

modes = nm.get("mode")
# modes = nm.mode

angles = np.zeros(nm.Nmodes)
for n,mode in enumerate(modes.T): # cycle over columns
    assert np.all(modes[:,n] == mode)
    angles[n] = angle_between_vectors(mode,direction)
angles *= 180 / np.pi
angles = np.mod(angles,180)
ic(angles.argmax(),": ",angles.max())

newmd = NormalModes(Nmodes=2,Ndof=nm.Ndof)
Q_D  = direction
Q_IR = nm.mode.isel(mode=26)
newmodes = np.asarray([Q_IR,Q_D]).T
newmd.set_modes(newmodes)
newmd.to_pickle("displ-mode.pickle")

density = 0.1 # dimensionless
IRamp = 0.5   # angstrom
QDexcess = 0.1 # angstrom
df = pd.DataFrame(columns=["mode", "start", "end",  "N"])
df["mode"] = [0,1]
df["start"] = [-IRamp,0-QDexcess]
df["end"] = [IRamp,dist+QDexcess]
df["N"]   = np.round(np.asarray(df["end"]-df["start"])/density).astype(int)

df.to_csv("displacements.csv",index=False)

pass

# displace-along-normal-modes.py -i ../start.ang.extxyz -nm displ-mode.pickle -d displacements.csv -o displaced-structures.ang.extxyz
# ovito displaced-structures.ang.extxyz
# convert-file.py -i displaced-structures.ang.extxyz -o displaced-structures.au.extxyz -iu angstrom -ou atomic_unit
# convert-file.py -i displaced-structures.ang.extxyz -o displaced-structures.au.xyz -iu angstrom -ou atomic_unit -of xyz