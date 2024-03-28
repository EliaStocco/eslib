import numpy as np
import matplotlib.pyplot as plt
from eslib.classes.trajectory import trajectory as Trajectory
from eslib.tools import convert
from eslib.classes.trajectory import info, array

trajectory = Trajectory("trajectory.extxyz")
time = info(trajectory,"time") * convert(1,"time","atomic_unit","femtosecond")

dipole = {
    "DFT" : info(trajectory,"dipoleDFT"),
     "LM" : info(trajectory,"DipoleModel"),
     "NN" : info(trajectory,"dipole")
}

# dipole
for k in dipole.keys():
    D = dipole[k]
    D -= D[0,:]
    argv = {
        "linestyle":"solid",
        "linewidth":1
    }
    fig,ax = plt.subplots(figsize=(10,4))
    ax.plot(time,D[:,0],label="$\\Delta$d$_x$",c="red",alpha=0.3,**argv)
    ax.plot(time,D[:,1],label="$\\Delta$d$_y$",c="blue",alpha=1,**argv)
    ax.plot(time,D[:,2],label="$\\Delta$d$_z$",c="green",alpha=0.3,**argv)
    ax2 = ax.twinx()

    # Efield
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

    ax.set_xlabel("time [fs]")
    ax.set_ylabel("dipole [q$_e$*bohr]")
    ax2.set_ylabel("E-field [eV/ang]")
    ax.grid()
    ax.set_title("dipole {:s}".format(k))
    ax.legend(title="dipole:",facecolor='white', framealpha=1,edgecolor="black",loc="upper left")
    ax2.legend(title="E-field:",facecolor='white', framealpha=1,edgecolor="black",loc="lower right")
    plt.tight_layout()
    plt.savefig("images/water.dipole-{:s}.pdf".format(k))