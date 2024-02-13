#!/usr/bin/env python
from eslib.classes import MicroState
from eslib.functions import get_one_file_in_folder
import numpy as np
import argparse
import os

description = ""

def main():

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-w", "--what"     , action="store", type=str,help="what to do: 's' (summary), 'j' (jmol)", default="js")
    parser.add_argument("-v", "--vib"      , action="store", type=str,help="folder with *.phonon.* files", default=".")
    parser.add_argument("-o", "--output"   , action="store", type=str,help="output file prefix", default="output")
    parser.add_argument("-q", "--positions", action="store", type=str,help="positions file (angstrom)",default=None)

    args = parser.parse_args()

    instructions = {
        "vib" : args.vib,
        # "positions" : "i-pi.pw-2y.positions_0.xyz",
        # "velocities" : "i-pi.pw-2y.velocities_0.xyz" ,
        # "relaxed"    : "start.au.xyz",
        # "properties"    : "i-pi.pw-2y.properties.out"
    }

    todo = "vib"
    data = MicroState(instructions=instructions,todo=todo)

    if 's' in args.what :
        df = data.vibrational_analysis_summary()
        # sprint(df)

        file = os.path.normpath("{:s}.csv".format(args.output))
        df.to_csv(file,index=False,fmt="%15.6f",sep=",",na_rep=np.nan)
    
    if 'j' in args.what:
        folder  = os.path.dirname(os.path.abspath(__file__))
        script  = os.path.normpath("{:s}/../eigvec-to-xyz_jmol.py".format(folder))
        pos     = args.positions
        modes   = get_one_file_in_folder(folder=args.vib,ext=".mode")
        eigvals = get_one_file_in_folder(folder=args.vib,ext=".eigval")
        file    = os.path.normpath("{:s}.jmol".format(args.output))
        cmd = "python {:s} {:s} {:s} {:s} {:s}".format(script,pos,modes,eigvals,file)
        os.system(cmd)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()

