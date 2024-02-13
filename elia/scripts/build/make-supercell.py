#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from elia.formatting import matrix2str
from ase.build import make_supercell
from scipy.spatial.transform import Rotation


#---------------------------------------#
description     = "Create a path bridgin two atomic structures (useful for NEB calculations)."
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
    parser.add_argument("-i" , "--input"        , **argv, type=str, help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-m" , "--matrix"       , **argv, type=str, help="txt file with the 3x3 transformation matrix")
    parser.add_argument("-o" , "--output"       , **argv, type=str, help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str, help="output file format (default: 'None')", default=None)
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
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=":")
    print("done")

    #-------------------#
    print("\tReading transformation matrix from file '{:s}' ... ".format(args.matrix), end="")
    matrix = np.loadtxt(args.matrix)
    print("done")

    # #-------------------#
    # print("\n\tDiagonalizing transformation matrix ... ", end="")
    # w,f = np.linalg.eig(matrix)
    # print("done")

    # print("\t eigenvalues: ",w)
    # print("\teigenvectors: ")
    # line = matrix2str(f.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    # print(line)

    # #-------------------#
    # print("\n\tExact decomposition of the transformation matrix:")
    # # Compute eigenvalues and eigenvectors
    # w, f = np.linalg.eig(matrix)
    # # compute D
    # s = np.absolute(w)
    # D = np.diag(s)
    # # compute R
    # R = np.linalg.inv(D) @ matrix
    # # Euler angles
    # rotation = Rotation.from_matrix(R)
    # # get Euler angles in radians
    # euler_angles_radians = rotation.as_euler('zyx')  # Order of rotation: 'zyx' (or any other order)
    # # convert radians to degrees if needed
    # euler_angles_degrees = np.degrees(euler_angles_radians)
    # print("\t\tstretching factors:", s)
    # print("\t\tEuler angles (deg):", euler_angles_degrees)

    # R = Rotation.from_euler('zyx', euler_angles_radians).as_matrix()

    # #-------------------#
    # print("\n\tRotating the atomic structures along the eigenvectors ... ", end="")
    # Rcell = [None] * len(atoms)
    # for n in range(len(atoms)):
    #     # Rpos  = (f @ atoms[n].get_positions().T).T
    #     Rcell[n] = (R @ atoms[n].get_cell()).T
    #     atoms[n].set_cell(Rcell[n].T,scale_atoms=True)
    #     #atoms[n].rotate(f,rotate_cell=True)
    # print("done")

    #-------------------#
    print("\n\tCreating the supercells ... ", end="")
    supercell = [None] * len(atoms)
    for n in range(len(atoms)):
        supercell[n] = make_supercell(atoms[n],matrix)
        # supercell[n].set_cell((matrix@Rcell[n].T).T)
    print("done")

    #-------------------#
    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=supercell,filename=args.output, format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

    #-------------------#
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
