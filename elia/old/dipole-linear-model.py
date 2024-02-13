#!/usr/bin/env python
from ase.io import read
import numpy as np
from ase import Atoms
# from elia.trajectory import info
from elia.classes.trajectory import trajectory as Trajectory
from elia.input import size_type

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate the dipole values according to a linear model\ngiven a reference configuration, and its BEC tensors and dipole."
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
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv,type=str, help="file with the atomic configurations [a.u]")
    parser.add_argument("-n", "--index"    , **argv,type=int, help="index of the reference configuration",default=None)
    parser.add_argument("-r", "--reference", **argv,type=str, help="file with the reference configuration [a.u.,xyz]")
    parser.add_argument("-d", "--dipole"   , **argv,type=lambda a:size_type(a,float,3), help="dipole of the reference configuration [a.u.] (default: None --> specify -n,--index)",default=None)
    parser.add_argument("-z", "--bec"      , **argv,type=str, help="file with the BEC tensors of the reference configuration [txt] (default: None --> specify -n,--index)",default=None)
    parser.add_argument("-f", "--frame"    , **argv,type=str, help="frame [eckart,global] (default: global)", default="global")
    parser.add_argument("-o", "--output"   , **argv,type=str, help="output file with the dipole values (default: 'dipole.linear.txt')", default="dipole.linear.txt")
    return parser.parse_args()

#---------------------------------------#
def dipole_linear_model(pos,ref,dipole,bec,frame="global"):
    match frame:
        case "eckart" :
            raise NotImplementedError("'eckart' frame not implemented yet.")
            from copy import copy
            newx, com, rotmat  = self.eckart(index)            
            # save old positions
            oldpos = copy(self.positions)
            # set the rotated positions
            self.positions = copy(newx.reshape((len(newx),-1)))
            # compute the model in the Eckart frame
            model, _, _ = self.dipole_model(index,frame="global")
            # re-set the positions to the original values
            self.positions = oldpos
            # return the model

            # 'rotmat' is supposed to be right-multiplied:
            # vrot = v @ rotmat
            return model, com, rotmat 

        case "global" :
            N = len(pos)
            model  = np.full((N,3),np.nan)
            for n in range(N):
                R = pos[n]#.reshape((-1,3))
                dD = bec.T @ (R - ref)
                model[n,:] = dD.flatten() + dipole
            return model, None, None
    
        case _ :
            raise ValueError("'frame' can be only 'eckart' or 'global' (dafault).")

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

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = Trajectory(args.input)
    print("done")

    #------------------#
    # index
    if args.index is not None:
        reference = trajectory[args.index]
        bec       = np.asarray(reference.arrays["bec"])
        dipole    = np.asarray(reference.info["dipole"])
        ref       = Atoms(reference)
        del reference

    else:
        raise NotImplementedError()

        #------------------#
        # reference
        print("\tReading reference structure from file '{:s}' ... ".format(args.reference), end="")
        reference = read(args.reference)
        print("done")

        #------------------#
        # BEC
        print("\tReading BEC tensor from file '{:s}' ... ".format(args.bec), end="")
        bec = np.loadtxt(args.bec)
        print("done")

    #------------------#
    # linear model
    print("\n\tComputing the dipole using a linear model ... ", end="")
    N        = len(trajectory)
    Natoms   = ref.get_positions().shape[0]
    pos      = np.asarray(trajectory.positions).reshape((N,-1))
    ref      = ref.get_positions().flatten()
    bec      = bec.reshape((Natoms*3,3))
    dipoleLM, _, _ = dipole_linear_model(pos,ref,dipole,bec,args.frame) 
    print("done")

    #------------------#
    # output
    print("\n\tSaving the dipoles to file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output,dipoleLM)
    print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()