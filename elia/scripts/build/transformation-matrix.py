#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
from elia.formatting import matrix2str
from elia.tools import find_transformation
from ase import Atoms
from ase.io import read
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

#---------------------------------------#
description     = "Compute the trasformation matrix M(A->B) between the lattice vector of the atomic configurations A and B."
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
def find_A2B(file_A,file_B):
    # structure A
    print("\tReading structure A from input file '{:s}' ... ".format(file_A), end="")
    A = read(file_A)
    print("done")

    print("\tCell A:")
    cell = np.asarray(A.cell).T
    line = matrix2str(cell.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    # structure B
    print("\tReading structure B from input file '{:s}' ... ".format(file_B), end="")
    B = read(file_B)
    print("done")

    print("\tCell B:")
    cell = np.asarray(B.cell).T
    line = matrix2str(cell.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    # trasformation
    M = find_transformation(A,B)
    print("\tTrasformation matrix M(A->B):")
    line = matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    det = np.linalg.det(M)
    print("\tdet(M): {:6.4f}".format(det))

    return M

#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a" , "--structure_A"  , type=str, **argv, help="atomic structure A [cell]")
    parser.add_argument("-b" , "--structure_B"  , type=str, **argv, help="atomic structure B [supercell]")
    parser.add_argument("-o" , "--output"       , type=str, **argv, help="output file for the trasformatio matrix", default=None)
    parser.add_argument("-of", "--output_format", type=str, **argv, help="output format for np.savetxt (default: '%%24.18e')", default='%24.18e')
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
    M = find_A2B(args.structure_A,args.structure_B)

    # M is the multiplication of a rotation matrix R and a diagonal matrix D
    # M = D@R --> R = D^{-1}@M 

    # #-------------------#
    # print("\n\tExact decomposition of the transformation matrix:")
    # # Compute eigenvalues and eigenvectors
    # w, f = np.linalg.eig(M)
    # # compute D
    # s = np.absolute(w)
    # D = np.diag(s)
    # # compute R
    # R = np.linalg.inv(D) @ M
    # # Euler angles
    # rotation = Rotation.from_matrix(R)
    # # get Euler angles in radians
    # euler_angles_radians = rotation.as_euler('zyx')  # Order of rotation: 'zyx' (or any other order)
    # # convert radians to degrees if needed
    # euler_angles_degrees = np.degrees(euler_angles_radians)
    # print("\t\tstretching factors:", s)
    # print("\t\tEuler angles (deg):", euler_angles_degrees)

    # #-------------------#
    # print("\n\tApproximate decomposition of the transformation matrix:")
    # # Compute eigenvalues and eigenvectors
    # w, f = np.linalg.eig(M)
    # # compute D
    # s = np.absolute(w).round()
    # D = np.diag(s)
    # # compute R
    # R = np.linalg.inv(D) @ M
    # # Euler angles
    # rotation = Rotation.from_matrix(R)
    # # get Euler angles in radians
    # euler_angles_radians = rotation.as_euler('zyx')  # Order of rotation: 'zyx' (or any other order)
    # # convert radians to degrees if needed
    # euler_angles_degrees = np.degrees(euler_angles_radians)
    # print("\t\tstretching factors:", s)
    
    # #-------------------#
    # print("\t\tapproximated trasformation matrix M(A->B):")
    # A = D @ Rotation.from_euler('zyx', euler_angles_radians).as_matrix()
    # line = matrix2str(A.round(2),col_names=["1","2","3"],cols_align="^",width=6)
    # print(line)

    #-------------------#
    if args.output is not None:
        print("\n\tSaving transformation matrix to file '{:s}' ... ".format(args.output),end="")
        np.savetxt(args.output,M,fmt=args.output_format)
        print("done")
    
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()