#!/usr/bin/env python
from ase.io import read
from scipy.spatial.distance import pdist, squareform
import numpy as np
import argparse
from eslib.classes.trajectory import trajectory as Trajectory
from tqdm import tqdm

#---------------------------------------#

# Description of the script's purpose
description = "Compute the maximum and minimum interatomic distances of a trajectory."
closure = "Job done :)"
input_arguments = "Input arguments"


#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

def prepare_args():

    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input",         **argv,type=str, help="input file")
    parser.add_argument("-if", "--input_format" , **argv,type=str, help="input file format (default: 'None')" , default=None)
    return parser.parse_args()

#---------------------------------------#
def main():

    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #---------------------------------------#
    # Read the XYZ file
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = Trajectory(args.input,format=args.input_format)
    print("done\n")

    #---------------------------------------#
    print("\tCompute interatormic distances ... ")
    max_distances = np.zeros(len(trajectory))
    Natoms = len(trajectory[0])
    out = np.full(int(Natoms*(Natoms-1)/2),np.nan)
    N = len(trajectory)
    with tqdm(enumerate(trajectory),bar_format='\t{l_bar}{bar:10}{r_bar}{bar:-10b}') as bar:
        for k,atoms in bar:

            # Get atomic positions
            positions = atoms.get_positions()

            # Compute pairwise distances
            distances = pdist(positions)

            # Convert the condensed distance matrix to a square matrix
            dist_matrix = squareform(distances)
            
            #print(out.shape)
            #break
            # Print the interatomic distances
            n = 0 
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    out[n] = dist_matrix[i,j]
                    n += 1
            #print(n)
            max_distances[k] = out.max()
    #---------------------------------------#
    print()
    print("\t{:>30s}: {:.4f}".format("min interatormic distance",max_distances.min()))
    print("\t{:>30s}: {:.4f}".format("max interatormic distance",max_distances.max()))
    print("\t{:>30s}: {:.4f}".format("delta distance",max_distances.max()-max_distances.min()))
        
    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
    
    





