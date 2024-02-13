#!/usr/bin/env python
import argparse
import numpy as np
from elia.tools import cart2lattice
from elia.classes.trajectory import trajectory

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dipole quanta."
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
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv,type=str, help="input file")
    parser.add_argument("-if", "--input_format" , **argv,type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-k"  , "--keyword"     , **argv,type=str, help="keyword (default: 'dipole')" , default="dipole")
    parser.add_argument("-o" , "--output"       , **argv,type=str, help="txt output file (default: 'quanta.txt')", default="quanta.txt")
    parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: '%%24.18e')", default='%24.18e')
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

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = trajectory(args.input,format=args.input_format)
    print("done")

    #---------------------------------------#
    # dipole
    print("\tExtracting '{:s}' from the trajectory ... ".format(args.keyword), end="")
    dipole = atoms.call(lambda e:e.info[args.keyword])
    print("done")

    #---------------------------------------#
    # lattice vectors
    print("\tExtracting the lattice vectors from the trajectory ... ", end="")
    lattices = atoms.call(lambda e:e.get_cell())
    print("done")

    #---------------------------------------#
    # pbc
    pbc = atoms.call(lambda e:e.pbc)
    if not np.all(pbc):
        raise ValueError("the system is not periodic: it's not")

    #---------------------------------------#
    # quanta
    print("\tComputing the dipole quanta ... ", end="")
    N = len(atoms)
    quanta = np.zeros((N,3))
    for n in range(N):
        cell = lattices[n].T
        R = cart2lattice(lattices[n])
        lenght = np.linalg.norm(cell,axis=0)
        quanta[n,:] = R @ dipole[n] / lenght
    print("done")

    #---------------------------------------#
    # output
    print("\n\tWriting dipole quanta to file '{:s}' ... ".format(args.output), end="")
    try:
        np.savetxt(args.output,quanta,fmt=args.output_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()

