#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------#
# Description of the script's purpose
description = "Plot a time series from a txt file."
warning = "***Warning***"
closure = "Job done :)"
error = "***Error***"
keywords = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"
#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN  + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input", type=str, help='input txt file')
    parser.add_argument("-o","--output", type=str, help='output file for the plot')
    return parser.parse_args()

#---------------------------------------#
def plot_array(input_file, output_file):
    # Load the numpy array from the input file
    data = np.atleast_2d(np.loadtxt(input_file))

    # Get the number of rows and columns from the array shape
    rows, cols = data.shape
    if rows == 1:
        data = data.T
        cols = rows

    # Create a plot for each row in the array
    fig,ax = plt.subplots(figsize=(15,5))
    for n in range(cols):
        ax.plot(data[:,n], label=str(n+1))

    # Add labels and legend
    plt.xlabel('time/row index')
    plt.grid()
    plt.tight_layout()
    plt.legend()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

#---------------------------------------#
def main():

    #------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    # Call the function with the provided arguments
    plot_array(args.input, args.output)

    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
