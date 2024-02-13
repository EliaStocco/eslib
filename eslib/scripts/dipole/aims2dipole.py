#!/usr/bin/env python
import re
import numpy as np
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Extract the values of the dipole from a file written by FHI-aims and convert to atomic_unit."
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA  + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    information     = Fore.YELLOW   + Style.NORMAL + information             + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"         , **argv,type=str, help="input txt file")
    parser.add_argument("-o", "--output"        , **argv,type=str, help="output file with the dipole values (default: 'dipole.aims.txt')", default="dipole.aims.txt")
    parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: '%%24.18f')", default='%24.18f')
    return parser.parse_args()

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

    #------------------#
    # Open the input file for reading and the output file for writing
    factor = convert(1,"length","angstrom","atomic_unit")
    n = 0 
    pattern = re.compile(r'Total dipole moment.*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?).*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?).*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)')
    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        # Iterate through each line in the input file
        for line in input_file:
            if "Total dipole moment" in line and "[eAng]" in line:
                # Search for the pattern and extract the first three float values
                # matches = re.search(r"Total dipole moment \[eAng\]\s*:\s*([-+]?\d*\.\d+|\d+\.\d*|\d+)", line)
                
                # If the pattern is found, extract and write the values to the output file
                # line = line.replace("E","e")
                matches = re.search(pattern, line)
        
                # If the pattern is found, extract and return the first three float values
                if matches:
                    float_values = [float(match) for match in matches.groups()[:3]]
                    float_values = np.asarray(float_values).reshape((1,3))
                    np.savetxt(output_file,factor*float_values,fmt=args.output_format)
                    n += 1 
                    # output_file.write(','.join(map(str, float_values)) + '\n')  # Save the values as a comma-separated line
    print("\tN. of dipole values found: ",n)

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()