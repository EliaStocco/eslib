#!/usr/bin/env python
import numpy as np
import os
from tqdm import tqdm
from elia.nn.functions import get_model
from elia.functions import suppress_output
from elia.input import str2bool
from elia.classes.trajectory import trajectory as Trajectory

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dipole values for a trajectory using a neural network"
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
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--instructions", type=str     , **argv, help="model input file (default: 'instructions.json')", default="instructions.json")
    parser.add_argument("-p" , "--parameters"  , type=str     , **argv, help="torch parameters file (default: 'parameters.pth')", default="parameters.pth",)
    parser.add_argument("-t" , "--trajectory"  , type=str     , **argv, help="trajectory file [a.u.]")
    parser.add_argument("-z" , "--compute_BEC" , type=str2bool, **argv, help="whether to compute BECs (default: false)", default=False)
    parser.add_argument("-d" , "--debug"       , type=str2bool, **argv, help="debug mode (default: false)", default=False)
    parser.add_argument("-o" , "--output"      , type=str     , **argv, help="output file with the dipoles (default: 'dipole.nn.txt')", default="dipole.nn.txt")
    parser.add_argument("-oz", "--output_BEC"  , type=str     , **argv, help="output file with the BECs (default: 'bec.nn.txt')", default="bec.nn.txt")
    return parser.parse_args()

#---------------------------------------#
def main():

    #---------------------------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.trajectory), end="")
    trajectory = Trajectory(args.trajectory)
    print("done")

    N = len(trajectory)
    print("\tn. of atomic structures: {:d}".format(N))

    #------------------#
    print("\tLoading model ... ",end="")
    file_in = os.path.normpath("{:s}".format(args.instructions))
    file_pa = os.path.normpath("{:s}".format(args.parameters))
    with suppress_output(not args.debug):
        model = get_model(file_in,file_pa)
        model.store_chemical_species(atoms=trajectory[0])
    print("done")

    #------------------#
    line = " and BEC" if args.compute_BEC else ""
    print("\tComputing dipole{:s} ... ".format(line))
    D = np.full((N,3),np.nan)
    if args.compute_BEC:
        Z = np.full((N,3*len(trajectory[0].positions),3),np.nan)

    #with tqdm(enumerate(trajectory)) as bar:
    for n,atoms in enumerate(trajectory):
        print("\t{:>6d}/{:<d}".format(n+1,N),end="\r")
        pos = atoms.positions
        cell = np.asarray(atoms.cell).T if np.all(atoms.get_pbc()) else None
        if args.compute_BEC:
            d,z,x = model.get_value_and_jac(pos=pos.reshape((-1,3)),cell=cell)
            Z[n,:,:] = z.detach().numpy()#.flatten()
        else:
            d,x = model.get(pos=pos.reshape((-1,3)),cell=cell)
        D[n,:] = d.detach().numpy()#.flatten()
        
    #------------------#
    print("\tSaving dipoles to file '{:s}' ... ".format(args.output),end="")
    np.savetxt(args.output,D)
    print("done")

    #------------------#
    if args.compute_BEC:
        print("\tSaving BECs to file '{:s}' ... ".format(args.output_BEC),end="")
        with open(args.output_BEC,"w") as f:
            for n in range(N):
                np.savetxt(f,Z[n,:,:],header="step: {:d}".format(n))
        print("done")
    

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()



print()