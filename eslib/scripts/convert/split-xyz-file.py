#!/usr/bin/env python
import glob
import os
import subprocess
import numpy as np
from eslib.formatting import esfmt, warning, error
from eslib.io_tools import count_lines, read_Natoms_homogeneous

#---------------------------------------#
# Description of the script's purpose
description = "Split an (ext)xyz file."

#---------------------------------------#
def prepare_args(description):
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , required=True ,**argv,type=str, help="xyz input file")
    parser.add_argument("-n" , "--number"       , required=True ,**argv,type=int, help="number of structures per file")
    parser.add_argument("-f" , "--folder"       , required=False,**argv,type=str, help="output folder (default: splitted)", default='splitted')
    parser.add_argument("-o" , "--output"       , required=True ,**argv,type=str, help="output prefix")
    parser.add_argument("-s" , "--suffix"       , required=False,**argv,type=str, help="suffix (default: xyz)", default='xyz')
    return parser

#---------------------------------------#
def verify_split(input_file, output_prefix, num_splits, lines_per_snapshot):
    """
    Verify the split files.

    Args:
        input_file (str): The path to the original input file.
        output_prefix (str): The prefix used for the split files.
        suffix (str): The suffix used for the split files.
        num_splits (int): The number of split files.
        lines_per_snapshot (int): The number of lines per snapshot.
    """
    total_lines = 0
    n_files = 0
    for split_file in glob.glob(output_prefix+"*"):
        # split_file = f"{output_prefix}{str(i).zfill(2)}{suffix}"
        split_lines = count_lines(split_file)
        #print(f'{split_file}: {split_lines} lines')
        total_lines += split_lines
        if split_lines % lines_per_snapshot != 0:
            print(f"\t - {warning}: {split_file} does not have lines proportional to {lines_per_snapshot}.")
        n_files += 1

    # print()
    if n_files != num_splits:
        print(f"\t - {error}: number of split files ({n_files}) does not match the expected number ({num_splits}).")
    else:
        print(f"\t - Success: number of split files matches the expected number.")

    # Count the lines in the original file
    original_lines = count_lines(input_file)

    if total_lines != original_lines:
        print(f"\t - {error}: total lines in split files ({total_lines}) do not match original file ({original_lines}).")
    else:
        print(f"\t - Success: total lines in split files match the original file.")

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    args.suffix = str("." + args.suffix).replace('..', '.')

    Natoms = read_Natoms_homogeneous(args.input)
    print(f'\n\tNumber of atoms: {Natoms}')
    
    # Count the number of lines in the input file
    print("\n\t Reading number of lines from file '{:s}' ... ".format(args.input), end="")
    total_lines = count_lines(args.input)
    print("done")
    print(f'\n\tTotal lines in the file: {total_lines}')

    # Compute the number of lines per split file    
    lines_per_snapshot = Natoms + 2
    print(f'\tLines per snapshot: {lines_per_snapshot}')
    lines_per_file = args.number * lines_per_snapshot
    num_splits = total_lines // lines_per_file
    if total_lines % lines_per_file != 0:
        num_splits += 1

    print(f'\tNumber of splits: {num_splits}')
    print(f'\tLines per split file: {lines_per_file}')
    suffix_length = int(np.ceil(np.log10(num_splits+1)))
    print(f'\tSuffix length: {suffix_length}')

    if args.folder is not None and args.folder not in ['', ' ']:
        os.makedirs(args.folder, exist_ok=True)
        args.output = os.path.join(args.folder, args.output)+"."

    # Construct the split command
    split_command = [
        'split',
        '-l', str(lines_per_file),
        '-d', f'--additional-suffix={args.suffix}',
        f'--suffix-length={suffix_length}',
        args.input,
        args.output
    ]
    
    cmd = ' '.join(split_command)
    print(f'\n\tRunning the following command:\n\t{cmd}')

    print("\n\tSplitting file '{:s}' into {:d} parts ... ".format(args.input, num_splits), end="")
    # Execute the split command
    subprocess.run(split_command, check=True)
    print("done")
    # print(f'\n\tFile has been split into {num_splits} parts.')

    # Verify the split files
    print("\n\tVerifying the split files:")
    verify_split(args.input, args.output, num_splits, lines_per_snapshot)

if __name__ == "__main__":
    main()
