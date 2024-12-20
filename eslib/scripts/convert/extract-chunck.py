#!/usr/bin/env python
import subprocess
from eslib.formatting import esfmt
from eslib.io_tools import read_Natoms_homogeneous, count_lines
from input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Extract the n^th chunck of a given length from an (ext)xyz file."

#---------------------------------------#
def prepare_args(description):
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , required=True ,**argv,type=str     , help="xyz input file")
    parser.add_argument("-l" , "--lines"        , required=True ,**argv,type=str2bool, help="read number of lines (default: %(default)s)", default=True)
    parser.add_argument("-n" , "--index"        , required=True ,**argv,type=int     , help="index of the chunck to print to file")
    parser.add_argument("-N" , "--chunck_size"  , required=True ,**argv,type=int     , help="chunck size")
    parser.add_argument("-o" , "--output"       , required=True ,**argv,type=str     , help="output file")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    print("\n\t Reading number of atoms from file '{:s}' ... ".format(args.input), end="")
    Natoms = read_Natoms_homogeneous(args.input)
    print("done")
    assert Natoms is not None, "An error occurred while reading the number of atoms from the file."
    print(f'\t Number of atoms: {Natoms}')
        
    #------------------#
    if args.lines:
        print("\n\t Reading number of lines from file '{:s}' ... ".format(args.input), end="")
        Nlines = count_lines(args.input)
        print("done")
        assert Nlines is not None, "An error occurred while reading the number of lines from the file."
        print(f'\t Number of lines: {Nlines}')
    
    #------------------#
    chunck_size = args.chunck_size*(Natoms+2)
    line_start = args.index*chunck_size
    line_end = (args.index+1)*chunck_size
    print(f'\t Chunck range (line numbers): {line_start} - {line_end}')
    if args.lines:
        if line_end > Nlines:
            raise ValueError(f"The chunck size ({args.chunck_size}) is larger than the number of lines in the file ({Nlines})")
        
    #------------------#
    # Use sed to extract the chunk of lines directly from the file and save to the output
    print("\n\t Extracting chunk using 'sed':")
    sed_command = f"sed -n '{line_start+1},{line_end+1}p' {args.input} > {args.output}"
    print(f"\t {sed_command}")
    subprocess.run(sed_command, shell=True, check=True)
    
    print(f'\n\t Chunk {args.index} with lines {line_start} - {line_end} has been written to {args.output}')
        
#---------------------------------------#
if __name__ == "__main__":
    main()
