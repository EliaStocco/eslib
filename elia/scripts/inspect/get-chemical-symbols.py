#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
from ase.io import read
import argparse
import json

#---------------------------------------#
description     = "Save to file and/or print to screen the chemical symbols of an atomic configuration."
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
        "-o", "--output",  type=str,**argv,
        help="output file", default=None
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

    symbols = atoms.get_chemical_symbols()

    print("\tAtomic symbols: ",end="")
    print("[",end="")
    N = len(symbols)
    for n,s in enumerate(symbols):
        if n < N-1:
            print(" '{:2}',".format(s),end="")
        else:
            print(" '{:2}'".format(s),end="")
    print("]")

    if args.output is not None:
        print("\n\tWriting atomic symbols to file '{:s}' ... ".format(args.output), end="")
        with open(args.output,"w") as f:
            data = {"chemical-symbols":symbols}
            json.dump(data, f)
        print("done")
    
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()