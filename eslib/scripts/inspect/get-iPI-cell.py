#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
from ase.io import read
import argparse
import numpy as np
from eslib.formatting import matrix2str

#---------------------------------------#
description     = "Save to file and/or print to screen the cell an atomic configuration in a i-PI compatible format."
warning         = "***Warning***"
closure         = "Job done :)"
keywords        = "It's up to you to modify the required keywords."
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
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-i", "--input",  type=str,**argv,
        help="atomic structure input file"
    )
    parser.add_argument(
        "-d", "--digits",  type=int,**argv,
        help="number of digits (default: 6)", default=6
    )
    parser.add_argument(
        "-e", "--exponential",  type=bool,**argv,
        help="exponential notation (default: False)", default=False
    )
    parser.add_argument(
        "-o", "--output",  type=str,**argv,
        help="output file (default: None)", default=None
    )
    options = parser.parse_args()
    return options

#---------------------------------------#
def main():

    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    # structure A
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    pbc = np.any(atoms.get_pbc())

    if not pbc:
        print("\n\tThe system is not periodic: providing a huge cell")
        cell = np.eye(3)*1e8
    else:
        cell = np.asarray(atoms.cell).T

    print("\tCell:")
    line = matrix2str(cell,col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    print("\tCell in the i-PI compatible format:")
    exp = "e" if args.exponential else "f" 
    num_format = f'{{:.{args.digits}{exp}}}'
    text_format = "[ " + (num_format+", ")*8 + num_format + "]"
    text = text_format.format(*cell.flatten().tolist())
    print("\t"+text)
    
    if args.output is not None:
        print("\n\tWriting cell in the i-PI compatible format to file '{:s}' ... ".format(args.output), end="")
        # Open a file in write mode ('w')
        with open(args.output, 'w') as file:
            # Write a string to the file
            file.write(text)
        print("done")
    
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()