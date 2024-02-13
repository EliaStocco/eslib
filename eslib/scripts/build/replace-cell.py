#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from eslib.input import str2bool


#---------------------------------------#
description     = "Replace the cell of a structure with the cell of another one, by keeping the fractional coordinates fixed."
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
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a", "--structure_A",  type=str,**argv,help="atomic structure A with the cell to be replaced")
    parser.add_argument("-b", "--structure_B",  type=str,**argv,help="atomic structure B with the replacing cell")
    parser.add_argument("-s", "--scale"      ,  type=str2bool,**argv,help="whether to rescale the positions", default=False)
    parser.add_argument("-o", "--output"     ,  type=str,**argv,help="output file")
    options = parser.parse_args()
    return options

#---------------------------------------#
def main():

    #-------------------#
    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #-------------------#
    print("\tReading atomic structure A from file '{:s}' ... ".format(args.structure_A), end="")
    A = read(args.structure_A,index=0)
    print("done")

    #-------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B,index=0)
    print("done")

    if not np.all(B.get_pbc()):
        raise ValueError("The structure that should provide the cell to be replace is not periodic.")
    
    #-------------------#
    if np.allclose(A.get_cell(),B.get_cell()):
        print("\n\t{:s}: the cell of the provided structures are already the same.".format(warning))

    #-------------------#
    if args.output is not None:

        #-------------------#
        print("\n\tReplaing the cell ... ", end="")
        cell   = B.get_cell()
        # scale = np.all(A.get_pbc())
        A.set_cell(cell,scale_atoms=args.scale)
        print("done")

        print("\n\tSaving the path to file '{:s}' ... ".format(args.output), end="")
        try:
            write(images=A,filename=args.output) # fmt)
        except Exception as e:
            print("\n\tError: {:s}".format(e))
        print("done")

    #-------------------#
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
