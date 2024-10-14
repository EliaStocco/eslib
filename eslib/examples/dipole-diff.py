import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.atomic_structures import info, array
from eslib.functions import plot_bisector
from eslib.tools import convert

trajectory = AtomicStructures.from_file("trajectory.extxyz")
time = info(trajectory,"time") * convert(1,"time","atomic_unit","femtosecond")
dipole = {
    "DFT" : info(trajectory,"dipoleDFT"),
     "LM" : info(trajectory,"DipoleModel"),
     "NN" : info(trajectory,"dipole")
}

NN_DFT = np.linalg.norm(dipole["NN"]-dipole["DFT"],axis=1)
DFT_LM = np.linalg.norm(dipole["LM"] - dipole["DFT"],axis=1)
NN_LM = np.linalg.norm(dipole["NN"] - dipole["LM"],axis=1)


# dipole
argv = {
    "linestyle":"solid",
    "linewidth":1,
    "alpha" : 0.7
}
fig,ax = plt.subplots(figsize=(10,4))
ax.plot(time,NN_DFT,label="|$\\mathbf{d}_{NN}$-$\\mathbf{d}_{DFT}$|",c="red",**argv)
ax.plot(time,DFT_LM,label="|$\\mathbf{d}_{LM}$-$\\mathbf{d}_{DFT}$|",c="blue",**argv)
#ax.plot(time,NN_LM ,label="|$\\mathbf{d}_{NN}$-$\\mathbf{d}_{LM}$|" ,c="green",**argv)
ax2 = ax.twinx()

# Efield
au2eVa = convert(1,"electric-field","atomic_unit","ev/a")
Efield = au2eVa * info(trajectory,"Efield")
A = np.max(Efield)
argv = {
    "linestyle":(5, (10, 3)),
    "linewidth":0.8,
    "alpha":0.3
}
ax2.plot(time,A*info(trajectory,"Eenvelope"),c="red",label="E$_{env}$",**argv)
ax2.plot(time,Efield[:,1],label="E$_{:s}$".format(["x","y","z"][1]),**argv)

ax.set_xlabel("time [fs]")
ax.set_ylabel("dipole [q$_e$*bohr]")
ax2.set_ylabel("E-field [eV/ang]")
ax.grid()
ax.legend(title="Dipole:",facecolor='white', framealpha=1,edgecolor="black",loc="upper left")
ax2.legend(title="E-field:",facecolor='white', framealpha=1,edgecolor="black",loc="lower right")
plt.tight_layout()
plt.savefig("images/water.cross-check.pdf")