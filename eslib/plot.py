from eslib.functions import add_default
import numpy as np

def straigh_line(ax,shift,get_lim,func,set_lim,**argv):

    default = {"color": "black", "alpha": 0.5, "linestyle": "dashed"}
    argv = add_default(argv,default)

    xlim = get_lim()
    
    func(shift,xlim[0],xlim[1],**argv)

    set_lim(xlim[0],xlim[1])

    return ax

def hzero(ax, shift=0, **argv):
    return straigh_line(ax,shift,ax.get_xlim,ax.hlines,ax.set_xlim,**argv)

def vzero(ax, shift=0, **argv):
    return straigh_line(ax,shift,ax.get_ylim,ax.vlines,ax.set_ylim,**argv)

    # default = {"color": "black", "alpha": 0.5, "linestyle": "dashed"}
    # argv = add_default(argv,default)

    # xlim = ax.get_xlim()
    
    # ax.hlines(shift,xlim[0],xlim[1],**argv)

    # return ax


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


def plot_bisector(ax, shiftx=0, shifty=0, **argv):
    default = {"color": "black", "alpha": 0.5, "linestyle": "dashed"}
    argv = add_default(argv,default)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # x1 = min(x.min(),y.min())
    # y2 = max(x.max(),y.max())
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