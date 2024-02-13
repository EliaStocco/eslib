import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_tensor(in_file, out_file):
    # Read the tensor from the file
    tensor = np.loadtxt(in_file)

    # Get the dimensions of the tensor
    rows, cols = tensor.shape

    # Create a figure and axis
    fig = plt.figure(figsize=(cols + 1.5, rows + 1))

    # Increase the width ratio for the last column to make it slightly bigger on the right
    gs = gridspec.GridSpec(rows + 1, cols + 2, width_ratios=[1] * cols + [0.2, 0.1], height_ratios=[1] * rows + [0.1])

    # Plot the tensor using imshow
    ax = plt.subplot(gs[:, :-2])
    im = ax.imshow(tensor, cmap='bwr', vmin=-np.max(np.abs(tensor)), vmax=np.max(np.abs(tensor)))

    # Add the values inside the cells
    for i in range(rows):
        for j in range(cols):
            value = f'{tensor[i, j]:.2f}'
            ax.text(j, i, value, color='black',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))

    # Add a colorbar for reference
    cax = plt.subplot(gs[:, -2])
    cbar = plt.colorbar(im, cax=cax)
    # cbar.set_label()

    # Add vertical and horizontal lines at integer numbers
    for i in range(cols + 1):
        ax.axvline(x=i - 0.5, color='black', linewidth=0.5)

    for i in range(rows + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=0.5)

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Move x-axis tick labels to the top
    ax.xaxis.tick_top()

    # Add custom x ticks (on top)
    ax.set_xticks([0, cols // 2, cols - 1])
    ax.set_xticklabels(["$\\partial$ d$^x$", "$\\partial$ d$^y$", "$\\partial$ d$^z$"])

    # Add custom y ticks and labels
    y_ticks = np.arange(rows)
    y_labels = [f'$\\partial$ R$^{{1}}_{{{axis}}}$' for axis in ['x', 'y', 'z'] for _ in range(rows // 3)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Save the figure
    plt.savefig(out_file)
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Visualize a tensor from a text file using imshow.')

    # Add the file path argument
    parser.add_argument("-i","--input", type=str, help='input file with the BEC tensor.')
    parser.add_argument("-o","--output", type=str, help='output file with the plot.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the plot_tensor function with the specified file path
    plot_tensor(args.input,args.output)
