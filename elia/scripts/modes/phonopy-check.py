#!/usr/bin/env python
from elia.classes.normal_modes import NormalModes
from elia.formatting import matrix2str
from elia.tools import convert
from elia.output import output_folder
from elia.input import size_type
from elia.functions import phonopy2atoms
import argparse
import numpy as np
import yaml
import pandas as pd
from icecream import ic
import os
# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
THRESHOLD = 1e-4
#---------------------------------------#
# Description of the script's purpose
description = "Check the integrity of a phonopy 'qpoints.yaml' file."
warning = "***Warning***"
closure = "Job done :)"
keywords = "It's up to you to modify the required keywords."
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
def prepare_args():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-q",  "--qpoints",       type=str, **argv, 
                        help="qpoints file (default: 'qpoints.yaml')", default="qpoints.yaml")
    # parser.add_argument("-i",  "--input",         type=str, **argv, 
    #                     help="general phonopy file (default: 'phonopy.yaml')", default="phonopy.yaml")
    # parser.add_argument("-o",  "--output",        type=str, **argv, 
    #                     help="output file (default: 'phonon-modes.pickle')", default="phonon-modes.pickle")
    # parser.add_argument("-of", "--output_folder", type=str, **argv, 
    #                     help="output folder for *.mode, *.eigvec and *.eigval files (default: None)", default=None)
    # parser.add_argument("-f", "--factor", type=float, **argv, 
    #                     help="conversion factor to THz for the frequencies ω", default=None)
    # parser.add_argument("-m",  "--matrices",      type=lambda x: size_type(x,str), **argv, 
    #                     help="matrices/vectors to print (default: ['eigval','eigvec','mode'])", default=['eigval','eigvec','mode'])
    return parser.parse_args()
#---------------------------------------#
def main():

    #---------------------------------------#
    # print the script's description
    print("\n\t{:s}".format(description))

    #---------------------------------------#
    # parse the user input
    args = prepare_args()
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    # #---------------------------------------#
    # # read input file ('phonopy.yaml')
    # print("\tReading data from input file '{:s}' ... ".format(args.input), end="")
    # with open(args.input) as f:
    #     info = yaml.safe_load(f)
    # print("done")

    # print("\t{:<10s} : ".format("dim"),info["phonopy"]["configuration"]["dim"])
    # print("\t{:<10s} : ".format("qpoints"),info["phonopy"]["configuration"]["qpoints"].split(" "))
    # print("\t{:<10s} : ".format("masses"),np.asarray([ a["mass"] for a in info["unit_cell"]["points"] ]).round(2))   

    # size = np.asarray([ int(a) for a in info["phonopy"]["configuration"]["dim"].split(" ") ])
    # factor = convert(1,"mass","dalton","atomic_unit")
    # mass = factor * np.asarray([ [a["mass"]]*3 for a in info["unit_cell"]["points"] ]).flatten()

    # #---------------------------------------#
    # # read input file ('phonopy.yaml')
    # print("\tExtracting reference atomic structure ... ", end="")
    # reference = phonopy2atoms(info["supercell"])
    # print("done")

    #---------------------------------------#
    # read qpoints file ('qpoints.yaml')
    print("\n\tReading qpoints from input file '{:s}' ... ".format(args.qpoints), end="")
    with open(args.qpoints) as f:
        qpoints = yaml.safe_load(f)
    print("done")
    
    print("\t{:<20s} : {:<10d}".format("# q points",qpoints["nqpoint"]))
    print("\t{:<20s} : {:<10d}".format("# atoms",qpoints["natom"]))
    print("\t{:<20s} :".format("reciprocal vectors"))
    tmp = np.asarray(qpoints["reciprocal_lattice"]).T
    line = matrix2str(tmp,digits=6,exp=False,width=12)
    print(line)

    #---------------------------------------#
    N = qpoints["natom"]*3
    nm = NormalModes(N,N)

    #---------------------------------------#
    for n in range(qpoints["nqpoint"]):
        phonon = qpoints["phonon"][n]
        q = tuple(phonon["q-position"])
        print("\t\tphonons {:d}: q point".format(n),q)
        
        nm.set_dynmat(phonon["dynamical_matrix"],mode="phonopy")
        nm.set_eigvec(phonon["band"],mode="phonopy")

        dynmat = nm.dynmat.to_numpy()
        eigvec = nm.eigvec.to_numpy()
        freq = np.asarray([ b["frequency"] for b in phonon["band"]])
        sorted_indices = np.argsort(freq)[::-1]
        freq = freq[sorted_indices]
        eigvec = eigvec[:, sorted_indices]

        if np.allclose(dynmat,eigvec):
            raise ValueError("'eigvec' and 'dynmat' are the same")
        else:
            norm = np.linalg.norm(dynmat-eigvec)
            print("\t\t|eigvec - dynmat| = {:4.2e}".format(norm))

            w,f = np.linalg.eigh(dynmat)
            # Get the indices that would sort the eigenvalues in descending order
            sorted_indices = np.argsort(w)[::-1]
            # Sort the eigenvalues and corresponding eigenvectors
            w = w[sorted_indices]
            f = f[:, sorted_indices]
            if not np.allclose(f,eigvec):
                norm = np.sqrt(np.square(f-eigvec).real.mean())
                print("\t\teigenvectors of 'dynmat' and 'eigvec' are not the same. Their average difference is {:4.2e}".format(norm))
    
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
