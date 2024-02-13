#!/usr/bin/env python
from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from ase.build import cut
from elia.tools import find_transformation
from elia.formatting import matrix2str

#---------------------------------------#
description     = "Determin which atom in structure A is a replica of an atom in structure B."
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
    parser.add_argument("-a", "--structure_A",  type=str,**argv,help="atomic structure A [cell]")
    parser.add_argument("-b", "--structure_B",  type=str,**argv,help="atomic structure B [supercell]")
    parser.add_argument("-o", "--output"     ,  type=str,**argv,help="output file for the trasformation matrix", default=None)
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
    A = read(args.structure_A)
    print("done")

    #-------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B)
    print("done")

    #-------------------#
    M = find_transformation(A,B)
    print("\n\tTrasformation matrix between cells of A and B:")
    line = matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    #-------------------#
    print("\n\tCreating a supercell of A ... ",end="")
    sA = make_supercell(A,[[2,0,0],[0,2,0],[0,0,2]])
    print("done")

    #-------------------#
    print("\n\tFinding replica ... ",end="")
    indices_A = []
    for atom_B in sA:
        # Find the minimum image convention
        indices, offset = cut(A.cell).index(atom_B.position)
        
        # The index in atoms_A corresponding to the atom in atoms_B
        index_A = A.get_number_of_atoms() * offset + indices[0]
        
        indices_A.append(index_A)
    print("done")

    from icecream import ic
    ic(indices_A)

    #-------------------#
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
