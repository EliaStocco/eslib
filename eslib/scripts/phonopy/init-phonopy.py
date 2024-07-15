#!/usr/bin/env python
from subprocess import run
import os
from eslib.formatting import esfmt
from eslib.input import ilist

#---------------------------------------#
# Description of the script's purpose
description = "Create the displaced structures using phonopy."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, type=str  , required=True , help="input file")
    parser.add_argument("-if", "--input_format" , **argv, type=str  , required=True , help="input file format")
    parser.add_argument("-t" , "--tolerance"    , **argv, type=float, required=False, help="symmetry tolerance (default: %(default)s)", default=1e-2)
    parser.add_argument("-s" , "--supercell"    , **argv, type=ilist, required=True ,  help="supercell")
    return parser

def create_phonopy_conf():
    with open("phonopy.conf", "w") as file:
        file.write("""
DIM = 1 1 1
EIGENVECTORS = .TRUE.
WRITE_DM = .TRUE.
QPOINTS = 0 0 0
""")


#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    cmd = f"phonopy --{args.input_format} -d --dim=\"{' '.join(map(str,args.supercell))}\" -c {args.input} -v --tolerance {args.tolerance}"
    print("\n\tRun the following command:")
    print("\t{:s}".format(cmd))
    
    cmd = f"phonopy -f lis_of_output_files"
    print("\t{:s}".format(cmd))

    cmd = f"phonopy -s phonopy.conf --tolerance {args.tolerance} -v --writedm >> phonopy.out"
    print("\t{:s}".format(cmd))

    create_phonopy_conf()

#---------------------------------------#
if __name__ == "__main__":
    main()
