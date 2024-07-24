from eslib.functions import add_default
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def generate_colors(N,map='tab10'):
    cmap = plt.get_cmap(map)  # You can choose other colormaps as well
    colors = [cmap(i) for i in np.linspace(0, 1, N)]
    return colors

def straigh_line(ax,shift,get_lim,func,set_lim,**argv):

    default = {"color": "black", "alpha": 0.5, "linestyle": "dashed"}
    argv = add_default(argv,default)

    xlim = get_lim()
    
    func(shift,xlim[0],xlim[1],**argv)

    set_lim(xlim[0],xlim[1])

    return ax

def hzero(ax, shift=0, **argv):
    """
    Plot a horizontal line on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        shift (float): The position of the line.
        **argv: Additional arguments to be passed to matplotlib.axes.Axes.hlines.

    Returns:
        matplotlib.axes.Axes: The modified axis.
    """
    return straigh_line(ax, shift, ax.get_xlim, ax.hlines, ax.set_xlim, **argv)


def vzero(ax, shift=0, **argv):
    """
    Plot a vertical line on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        shift (float): The position of the line.
        **argv: Additional arguments to be passed to matplotlib.axes.Axes.vlines.

    Returns:
        matplotlib.axes.Axes: The modified axis.
    """
    return straigh_line(ax, shift, ax.get_ylim, ax.vlines, ax.set_ylim, **argv)

def square_plot(ax,lims:tuple=None):
    if lims is None :
        x = ax.get_xlim()
        y = ax.get_ylim()

        l, r = min(x[0], y[0]), max(x[1], y[1])
    else:
        l,r = lims

    ax.set_xlim(l, r)
    ax.set_ylim(l, r)
    return ax

def plot_bisector(ax, shiftx=0, shifty=0, argv:dict=None):
    default = {"color": "black", "alpha": 0.5, "linestyle": "dashed"}
    argv = add_default(argv,default)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x1 = min(xlim[0], ylim[0])
    y2 = max(xlim[1], ylim[1])
    bis = np.linspace(x1, y2, 1000)

    ax.plot(bis + shiftx, bis + shifty, **argv)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return

def align_yaxis(ax1, ax2, v1=0, v2=0):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
    
def remove_empty_space(ax):
    """
    Adjusts the x-axis limits (xlim) of the given axis according to the minimum and maximum values encountered in the plotted data.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to adjust the x-axis limits for.
    """
    # Get the lines plotted on the axis
    lines = ax.get_lines()

    # Initialize min and max values with the first line's data
    min_x, max_x = lines[0].get_xdata().min(), lines[0].get_xdata().max()

    # Iterate over the rest of the lines to find the overall min and max values
    for line in lines[1:]:
        min_x = min(min_x, line.get_xdata().min())
        max_x = max(max_x, line.get_xdata().max())

    # Set the x-axis limits accordingly
    ax.set_xlim(min_x, max_x)

#---------------------------------------#
def histogram(data: np.ndarray, file: str=None):
    """
    Plot histograms for each column of the input array.

    Parameters:
        data (np.ndarray): Input array of shape (N, M).
        file (str): Output filename for the plot.

    Returns:
        None
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape((len(data),1))
    elif data.ndim > 2:
        data = data.reshape((len(data),-1))
    num_columns = data.shape[1]
    figsize = (6 * num_columns, 5)  # Adjust figsize based on the number of columns

    # bin_widths = []   # Store bin widths for each column

    # Plot histograms for each column
    plt.figure(figsize=figsize)

    for i in range(num_columns):
        column_data = data[:, i]
        
        # Freedman-Diaconis rule for bin width
        iqr = np.percentile(column_data, 75) - np.percentile(column_data, 25)
        bin_width = 2 * iqr / (len(column_data) ** (1/3))
        num_bins_fd = int(np.ceil((column_data.max() - column_data.min()) / bin_width))
        # bin_widths.append(bin_width)

        # Plot histogram for the current column
        plt.hist(column_data, bins=num_bins_fd, edgecolor='black', alpha=0.7, label=f'Column {i}')

    # Add legend and labels
    plt.title('Histograms of Columns')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    if file is not None:
        plt.savefig(file)
    else :
        plt.show()

#---------------------------------------#
def plot_matrix(M,Natoms=None,file=None):
    import matplotlib.pyplot as plt  
    # from matplotlib.colors import ListedColormap
    # Create a figure and axis
    fig, ax = plt.subplots()  
    argv = {
        "alpha":0.5
    }
    ax.matshow(M, origin='upper',extent=[0, M.shape[1], M.shape[0], 0],**argv)
    if Natoms is not None:
        argv = {
            "linewidth":0.8,
            "linestyle":'--',
            "color":"white",
            "alpha":1
        }
        xx = np.arange(0,M.shape[0],Natoms*3)
        yy = np.arange(0,M.shape[1],Natoms*3)
        for x in xx:
            ax.axhline(x, **argv) # horizontal lines
        for y in yy:
            ax.axvline(y, **argv) # horizontal lines
        
        

        xx = xx + np.unique(np.diff(xx)/2)
        N = int(np.power(len(xx),1/3)) # int(np.log2(len(xx)))
        ticks = list(product(*([np.arange(N).tolist()]*3)))
        ax.set_xticks(xx)
        ax.set_xticklabels([str(i) for i in ticks])
        # ax.xaxis.set(ticks=xx, ticklabels=[str(i) for i in ticks])
        
        yy = yy + np.unique(np.diff(yy)/2)
        N = int(np.power(len(yy),1/3))
        ticks = list(product(*([np.arange(N).tolist()]*3)))
        # ax.yaxis.set(ticks=yy, ticklabels=ticks)
        ax.set_yticks(yy)
        ax.set_yticklabels([str(i) for i in ticks])

    plt.tight_layout()
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
    return