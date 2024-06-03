#!/usr/bin/env python
import argparse
# import numpy as np
from ase.io import read
from ase import Atoms
import pandas as pd
# from icecream import ic
from eslib.classes.vectorize import easyvectorize
from eslib.classes.normal_modes import NormalModes
# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
# Description of the script's purpose
description = "Project a trajectory onto phonon modes."
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
    # parser.add_argument("-g",  "--ground_state", type=str, **argv, help="ground-state atomic structure [a.u.] (default: %(default)s)", default="start.xyz")
    parser.add_argument("-t",  "--trajectory",   type=str, **argv, help="input extxyz file [a.u.] (default: %(default)s)", default="trajectory.extxyz")
    parser.add_argument("-pm", "--phonon_modes", type=str, **argv, help="phonon modes file computed by 'post-process-phonopy.py' (default: %(default)s)", default="phonon-modes.pickle")
    parser.add_argument("-o",  "--output",       type=str, **argv, help="output file (default: %(default)s)", default="trajectory.phonon-modes.pickle")
    return parser# .parse_args()

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
    # read trajectory
    print("\tReading trajectory from file '{:s}' ... ".format(args.trajectory), end="")
    atoms = read(args.trajectory,format="extxyz",index=":")
    print("done")

    print("\tVectorizing the trajectory ... ", end="")
    trajectory = easyvectorize(Atoms)(atoms)
    del atoms
    print("done")

    #---------------------------------------#
    # read phonon modes ('phonon-modes.pickle')
    print("\tReading phonon modes from file '{:s}' ... ".format(args.phonon_modes), end="")
    pm = pd.read_pickle(args.phonon_modes)
    print("done")

    if type(pm) != pd.DataFrame:
        raise TypeError("Loaded object is of wrong type, it should be a 'pandas.DataFrame' object")

    #---------------------------------------#
    # project on phonon modes
    print("\n\tProjecting the trajectory to:")
    results = pd.DataFrame(index=pm.index,columns=["q"])
    k = 0
    for n,row in pm.iterrows():
        q = row["q"]
        print("\t\t- phonon modes {:d} with q-point {:s} ... ".format(k,str(q)), end="")
        if type(row["supercell"]) != NormalModes:
            raise TypeError("'supercell' element is of wrong type, it should be a 'NormalModes' object")     
        # row["modes"].write("test.yaml","yaml")   
        out = row["supercell"].project(trajectory)
        results.at[q,"q"] = q
        for c in out.keys():
            if c not in results.columns:
                results[c] = None
            results.at[q,c] = out[c]
        print("done")
        k += 1
        
    #---------------------------------------#
    # save result to file
    print("\n\tWriting results to file '{:s}' ... ".format(args.output), end="")
    results.to_pickle(args.output)
    print("done")

    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()