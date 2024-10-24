import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from eslib.functions import plot_bisector, square_plot

# Load data
A = np.loadtxt("../dipole.test-dft.txt")
B = np.loadtxt("../dipole.nn-test.txt")

# Create the main scatter plot
fig, ax = plt.subplots(figsize=(6, 5))

labels = ["x", "y", "z"]
ax.scatter(A[:, 0], B[:, 0], label="d$_x$", color="red", s=2)
ax.scatter(A[:, 1], B[:, 1], label="d$_y$", color="blue", s=2)
ax.scatter(A[:, 2], B[:, 2], label="d$_z$", color="green", s=2)
# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
plot_bisector(ax)
ax = square_plot(ax)

ax.grid()
ax.legend(title="Dipole:")

ax.set_xlabel("dipole DFT [q$_e\\times$bohr]")
ax.set_ylabel("dipole NN [q$_e\\times$bohr]")

# Create the inset correlation plot
axins = inset_axes(ax, width="30%", height="35%", loc="lower right", bbox_to_anchor=(0, 0.07, 1, 1), bbox_transform=ax.transAxes)
axins.scatter(np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1), label="|$\\mathbf{d}$|",color="black", s=2)
axins.legend(title="Dipole:")
#axins.set_xlabel("dipole DFT")
#axins.set_ylabel("dipole NN")
axins.grid()
#axins.set_title("Correlation Plot")
# axins.set_xlim(0.72,0.88)
# axins.set_ylim(0.72,0.88)
plot_bisector(axins)
axins = square_plot(axins)


# Save the combined plot
#plt.tight_layout()
plt.savefig("LiNbO3.corr.300K.pdf")
# plt.show()