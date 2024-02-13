import numpy as np
import matplotlib.pyplot as plt
from elia.classes.trajectory import trajectory as Trajectory
from elia.tools import convert
from elia.classes.trajectory import info, array

trajectory = Trajectory("trajectory.extxyz")
time = info(trajectory,"time") * convert(1,"time","atomic_unit","femtosecond")

labels = []
fig,ax = plt.subplots(figsize=(10,4))

# energy
argv = {
    "linestyle":"solid",
    "linewidth":1
}
au2meV = convert(1,"energy","atomic_unit","millielectronvolt")
Etot = au2meV * info(trajectory,"conserved")
Ekin = au2meV * info(trajectory,"kinetic_md") 
Epot = au2meV * info(trajectory,"potential") 
Etot -= Epot[0]
Epot -= Epot[0]
ax.plot(time,Etot,c="blue",label="E$_{tot}$",**argv)
ax.plot(time,Ekin,c="red",label="E$_{kin}$",alpha=0.3,**argv)
ax.plot(time,Epot,c="green",label="E$_{pot}$",alpha=0.3,**argv)

# Efield
ax2 = ax.twinx()
au2eVa = convert(1,"electric-field","atomic_unit","ev/a")
Efield = au2eVa * info(trajectory,"Efield")
A = np.max(Efield)
argv = {
    "linestyle":(5, (10, 3)),
    "linewidth":0.8,
    "alpha":0.6
}
ax2.plot(time,A*info(trajectory,"Eenvelope"),c="red",label="E$_{env}$",**argv)
ax2.plot(time,Efield[:,1],label="E$_{:s}$".format(["x","y","z"][1]),**argv)

# plot
ax.set_xlabel("time [fs]")
ax.set_ylabel("energy [meV]")
ax2.set_ylabel("E-field [eV/ang]")
ax.grid()
labs = [l.get_label() for l in labels]
# ax.legend(labels, labs)
ax.legend(title="Energy:",facecolor='white', framealpha=1,edgecolor="black")
ax2.legend(title="E-field:",facecolor='white', framealpha=1,edgecolor="black",loc="lower right")
#plt.title("energy and E-field")
plt.tight_layout()
#plt.show()
plt.savefig("images/water.energy-Efield.pdf")