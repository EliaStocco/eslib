#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
from eslib.formatting import esfmt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.cm import ScalarMappable

#---------------------------------------#
# Description of the script's purpose
description = "Plot the unit-cells dipoles."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"  , **argv, required=True , type=str, help="*.txt file with the results produced by 'unit-cell-dipole.py'")
    parser.add_argument("-N" , "--step"   , **argv, required=False, type=int, help="consider only every N-th snapshot (default: %(default)s)", default=1)
    parser.add_argument("-n" , "--indices", **argv, required=True , type=str, help="*.txt file with the indices produced by 'divide-into-unitcells-with-indices.py'")
    parser.add_argument("-k" , "--keyword", **argv, required=True , type=str, help="keyword to print")
    # parser.add_argument("-l" , "--limits", **argv, required=True , type=str, help="keyword to print")
    parser.add_argument("-o" , "--output" , **argv, required=True , type=str, help="output folder")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    os.makedirs(args.output,exist_ok=True)
    ofile = f"{args.output}/data4plot.csv"
    
    #------------------#
    if os.path.exists(ofile):
        print("\tReading data from '{:s}' ... ".format(ofile), end="")
        df = pd.read_csv(ofile)
        print("done")
    
    #------------------#
    else:

        #------------------#
        print("\tReading data from '{:s}' ... ".format(args.input), end="")
        df = pd.read_csv(args.input, sep='\s+')
        df["structure"] = df["structure"].astype(int)
        df["unit_cell"] = df["unit_cell"].astype(int)
        print("done")
        print("\tdf.shape: ",df.shape)
        
        #------------------#
        print("\tReading indices from '{:s}' ... ".format(args.input), end="")
        indices = np.loadtxt(args.indices, dtype=int)
        print("done")
        print("\tindices.shape: ",indices.shape)
        
        #------------------#
        indices = pd.DataFrame(data=indices,columns=["unit_cell","R_1","R_2","R_3"])
        indices.set_index("unit_cell",inplace=True)
        indices = indices[~indices.index.duplicated(keep='first')]

        Rxyz = indices.loc[np.asarray(df["unit_cell"],dtype=int)]
        for n in range(1,4):
            df[f"R_{n}"] = np.asarray(Rxyz[f"R_{n}"])

        #------------------#
        print("\tSaving data to '{:s}' ... ".format(ofile), end="")
        df.to_csv(ofile,index=False)
        print("done")

    #------------------#
    df["structure"] = df["structure"].astype(int)
    df["unit_cell"] = df["unit_cell"].astype(int)
    structures = np.unique(df["structure"])
    for n in range(len(structures)):
        
        if n % args.step != 0:
            continue
        
        pfile = f"{args.output}/cube.n={structures[n]}.png"
        print("\tPlotting data for structure {:d} to file {:s} ... ".format(structures[n],pfile), end="\r")
        
        sub_df = df[df["structure"] == structures[n]].copy()  # .copy() here

        del sub_df["structure"]
        del sub_df["unit_cell"]
        
        # tmp = sub_df.loc[:,["dipole_x","dipole_y","dipole_z"]].values
        # sub_df.loc[:, "dipole"] = np.linalg.norm(tmp, axis=1)
        # del sub_df["dipole_x"]
        # del sub_df["dipole_y"]
        # del sub_df["dipole_z"]
        
        N1 = len(np.unique(sub_df["R_1"]))
        N2 = len(np.unique(sub_df["R_2"]))
        N3 = len(np.unique(sub_df["R_3"]))
        
        data = np.full((N1,N2,N3),np.nan)
        for nx in range(N1):
            for ny in range(N2):
                for nz in range(N3):
                    value = sub_df.loc[
                        (sub_df["R_1"]==nx) & (sub_df["R_2"]==ny) & (sub_df["R_3"]==nz),
                        args.keyword
                    ].values
                    assert len(value) == 1, \
                        f"Number of values ({len(value)}) does not match the expected number (1)"
                        
                    data[nx,ny,nz] = float(value[0])
                    
        assert not np.any(np.isnan(data)), \
            f"Data contains NaN values. Please check the input data."
        
        
        # Create a figure and an axis for 3D plotting
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')

        # Create custom blue-white-red colormap
        cmap = colors.LinearSegmentedColormap.from_list("bwr_custom", ["blue", "white", "red"])

        # Normalize with midpoint (vcenter) at zero
        max_abs = np.max(np.abs(data))
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

        # Plot small cubes within the 3D space
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    # Get the data value at position (i, j, k)
                    value = data[i, j, k]
                    
                    # Calculate the color for this cube based on its value
                    color = cmap(norm(value))

                    # Create the position for each cube and plot it
                    ax.bar3d(i, j, k, 1, 1, 1, color=color, shade=True)

        

        # Remove ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        # Remove axis panes (backgrounds)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        # ax.set_xlabel(r"a$_1$")
        # ax.set_ylabel(r"a$_2$")
        # ax.set_zlabel(r"a$_3$")
        
        # Adjust padding (distance) from the axes
        ax.tick_params(axis='x', pad=0)  # For x-axis
        ax.tick_params(axis='y', pad=0)  # For y-axis
        ax.tick_params(axis='z', pad=0)  # For z-axis

        # Remove grid lines
        ax.grid(False)
        # Show the plot
        
        # Create a mappable for the colorbar
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])  # Dummy array to avoid warning
        # cbar = plt.colorbar(mappable, ax=ax, pad=0.1)    
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=r"dipole $\mu$ [e/$\mathrm{\AA}$]")
        
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(pfile,dpi=600,transparent=False,bbox_inches='tight')
        plt.close(fig)
    
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
    