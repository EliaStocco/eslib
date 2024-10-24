import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from sklearn.metrics import mean_squared_error, r2_score

from eslib.classes.atomic_structures import AtomicStructures, array, info
from eslib.classes.bec import bec as BEC
from eslib.plot import (align_yaxis, hzero, plot_bisector, remove_empty_space,
                        square_plot)
from eslib.tools import convert

fontsize = 16
Emodes = pd.read_csv("nm/energy.csv",header=None)
au2meV = convert(1,"energy","atomic_unit","electronvolt")
Emodes *= au2meV

trajectory = AtomicStructures.from_file("trajectory.extxyz")
time = info(trajectory,"time") * convert(1,"time","atomic_unit","femtosecond")

Etot = info(trajectory,"conserved")
Etot -= info(trajectory,"potential")[0]
Etot *= au2meV

fig,ax = plt.subplots(figsize=(10,4))

# Set font sizes
plt.rcParams.update({'font.size': fontsize})  # Increase font size for all text
# Set tick sizes
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1.5, length=6)  # Increase major tick size
ax.tick_params(axis='both', which='minor', width=1.5, length=4)  # Increase minor tick size


# Normal modes energy
argv = {
    "linestyle":"solid",
    "linewidth":1,
    "alpha":1.0
}
ax.plot(time,Emodes[6],label=r'E$_{\rm bend}$',c="firebrick",zorder=0,**argv)
ax.plot(time,Emodes[7],label=r'E$_{\rm str,sym}$',c="blue",zorder=0,**argv)
ax.plot(time,Emodes[8],label=r'E$_{\rm str,asym}$',c="green",zorder=0,**argv)

# Total energy and anharmonic contribution
argv = {
    "linestyle":"solid", #(5, (10, 3)),
    "linewidth":0.4,
    "alpha":1
}
Eharm = Emodes.sum(axis=1)
Eanharm = Etot - Eharm
ax.plot(time,Etot,c="black",label=r'E$_{\rm tot}^{\rm DFT}$',**argv)
ax.plot(time,Eanharm,c="purple",label=r'E$_{\rm anharm}$',**argv)

# Efield
ax2 = ax.twinx()
au2eVa = convert(1,"electric-field","atomic_unit","ev/a")
Efield = au2eVa * info(trajectory,"Efield")
A = np.max(Efield)
argv = {
    "linestyle":"solid", #(5, (10, 3)),
    "linewidth":0.7,
    "alpha":0.3
}
ax2.plot(time,A*info(trajectory,"Eenvelope"),c="firebrick",label=r'$\mathcal{E}_{\rm env}$',zorder=0,**argv)
ax2.plot(time,Efield[:,1],label=r'$\mathcal{E}$',zorder=0,color="blue",**argv)

#align_yaxis(ax,ax2)
remove_empty_space(ax)
argv = {
    "linestyle":"solid", #(5, (10, 3)),
    "linewidth":0.5,
    "alpha":1
}
hzero(ax2,**argv)
ax2.set_ylim(-7,7)
ax.set_ylim(-0.3,1.1)

ax.set_xlabel("time [fs]",fontsize=fontsize)
ax.set_ylabel("energy [eV]",fontsize=fontsize)
ax2.set_ylabel("E-field [eV/ang]",fontsize=fontsize)
ax.grid(linestyle="dashed",alpha=0.7)

# legend1 = ax2.legend(title="E-field:",facecolor='white', ncol=2, \
#            framealpha=1,edgecolor="black",loc="lower right",fontsize=fontsize)
# legend2 = ax.legend(title="Energy:",facecolor='white', ncol=5,\
#           framealpha=1,edgecolor="black",loc="lower left",fontsize=fontsize)

# Create proxy artists for the legend with thicker lines
linewidth=2
line1_proxy = Line2D([0], [0], color='firebrick', linewidth=linewidth)
line2_proxy = Line2D([0], [0], color='blue', linewidth=linewidth)
line3_proxy = Line2D([0], [0], color='green', linewidth=linewidth)
line4_proxy = Line2D([0], [0], color='black', linewidth=linewidth)
line5_proxy = Line2D([0], [0], color='purple', linewidth=linewidth)

# Move the legend outside the plot
legend1 = ax.legend([line1_proxy, line2_proxy, line3_proxy, line4_proxy, line5_proxy],
                    [r'E$_{\rm bend}$', r'E$_{\rm str,sym}$', r'E$_{\rm str,asym}$', 
                     r'E$_{\rm tot}^{\rm DFT}$', r'E$_{\rm anharm}$'],
                    title="Energy:", facecolor='white', ncol=5,
                    framealpha=1, edgecolor="black", loc="lower left", fontsize=fontsize-2, title_fontsize=fontsize-2)

# Create proxy artists for the legend with thicker lines
linewidth=2
line1_proxy = Line2D([0], [0], color='firebrick', linewidth=linewidth,alpha=0.3)
line2_proxy = Line2D([0], [0], color='blue', linewidth=linewidth,alpha=0.3)
# # Move the legend outside the plot
# legend1 = ax.legend(title="Energy:", facecolor='white', ncol=5,\
#           framealpha=1, edgecolor="black", loc="lower left", fontsize=fontsize-2,title_fontsize=fontsize-2)
legend2 = ax2.legend([line1_proxy, line2_proxy],[r'$\mathcal{E}_{\rm env}$',r'$\mathcal{E}$'],\
                     title="E-field:", facecolor='white', ncol=2, \
           framealpha=1, edgecolor="black", loc="lower right", fontsize=fontsize-1,title_fontsize=fontsize-2)

legend1._legend_box.align = "left"
legend2._legend_box.align = "left"


# Set the position of the legend
# ax.add_artist(legend1)
# legend1.set_bbox_to_anchor((-0.03, -0.50))  # Adjust the values as needed
# ax2.add_artist(legend2)
# legend2.set_bbox_to_anchor((-0.03, -0.50))  # Adjust the values as needed
ax.add_artist(legend1)
legend1.set_bbox_to_anchor((-0.01, -0.52))  # Adjust the values as needed
ax2.add_artist(legend2)
legend2.set_bbox_to_anchor((0.32, -0.77))  # Adjust the values as needed

plt.tight_layout()
# plt.show()
for ext in ["pdf"]: #,"png","jpg"
    plt.savefig(f'images/water.energy.time-series.{ext}',dpi=1200,bbox_inches='tight')