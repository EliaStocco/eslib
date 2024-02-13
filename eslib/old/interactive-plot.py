#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import argparse
from ase.io import read
from ase.visualize import view
from ase.visualize.external import viewers

#---------------------------------------#
description     = "Produce an interactive plot of an atomic structure."
warning         = "***Warning***"
closure         = "Job done :)"
keywords        = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"
divisor         = "-"*100

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
    divisor         = Fore.CYAN   + Style.NORMAL + divisor         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-i", "--input",  type=str,**argv,
        help="atomic structure input file"
    )
    parser.add_argument(
        "-v", "--viewer",  type=str,**argv,
        help="viewer {:s}".format(str(list(viewers.keys()))), default="ase"
    )
    return parser.parse_args()

#---------------------------------------#
def main():

    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))

    #---------------------------------------#
    print("\n\t{:s}".format(divisor))
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    # print(viewers.keys())
    #---------------------------------------#
    print("\tShowing atomic structure using the viewer '{:s}' ... ".format(args.viewer), end="")
    view(atoms,viewer=args.viewer,block=True)
    print("done")

    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()