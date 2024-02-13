#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from eslib.tools import segment


#---------------------------------------#
description     = "Create a path bridging two atomic structures (useful for NEB calculations)."
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
    parser.add_argument("-a", "--structure_A",  type=str,**argv,help="atomic structure A")
    parser.add_argument("-b", "--structure_B",  type=str,**argv,help="atomic structure B")
    parser.add_argument("-n", "--number"     ,  type=int,**argv,help="number of inner structures")
    parser.add_argument("-o", "--output"     ,  type=str,**argv,help="output file with the path", default="path.xyz")
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

    #-------------------#
    if not np.allclose(A.get_cell(),B.get_cell()):
        raise ValueError("The two structures do not have the same cell.")

    #-------------------#
    print("\n\tComputing the path positions ... ", end="")
    N=args.number
    pathpos = np.zeros((N + 2, *A.positions.shape))
    # pm = np.full(A.positions.shape,1)
    As = A.get_scaled_positions()
    Bs = B.get_scaled_positions()
    As[( As - Bs ) > +0.5] -= 1
    As[( As - Bs ) < -0.5] += 1
    T = np.linspace(0, 1, N + 2)
    # N = 0 -> t=0,1
    # N = 1 -> t=0,0.5,1
    for n, t in enumerate(T):
        # t = float(n)/(N+1)
        pathpos[n] = As * (1 - t) + t * Bs
    print("done")

    #-------------------#
    N = pathpos.shape[0]
    print("\tn. of structures in the path: '{:d}'".format(N))

    #-------------------#
    print("\tCreating the path ... ", end="")
    path = [None]*N
    for n in range(N):
        path[n] = A.copy()
        path[n].set_scaled_positions(pathpos[n])
    print("done")

    #-------------------#
    print("\n\tSaving the path to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=path,filename=args.output) # fmt)
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    print("done")

    #-------------------#
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
