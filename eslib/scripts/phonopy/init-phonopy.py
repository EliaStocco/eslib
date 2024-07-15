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
    # parser.add_argument("-o" , "--output"       , **argv, type=str  , required=False, help="output file (default: %(default)s)", default="init-phonopy.txt")
    # parser.add_argument("-f" , "--folder"       , **argv, type=str  , required=False , help="folder where all files will be created (default: %(default)s)", default="init-phonopy")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    # cwd = os.getcwd()
    # os.makedirs(args.folder, exist_ok=True)
    # os.chdir(args.folder)

    # files = os.listdir(".")
    # for file in files:
    #     os.remove(file)

    cmd = f"phonopy --{args.input_format} -d --dim=\"{' '.join(map(str,args.supercell))}\" -c {args.input} -v --tolerance {args.tolerance}"
    print("\n\tRun the following command:")
    print("\t{:s}".format(cmd))
    # run(cmd.split(" "), check=True)

    # os.chdir(cwd) #come back to the original folder

#---------------------------------------#
if __name__ == "__main__":
    main()