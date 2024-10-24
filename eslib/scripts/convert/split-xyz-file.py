#!/usr/bin/env python
import glob
import os
import subprocess

from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Split a xyz file."

#---------------------------------------#
def prepare_args(description):
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , required=True ,**argv,type=str, help="xyz input file")
    parser.add_argument("-a" , "--atoms"        , required=True ,**argv,type=int, help="number of atoms")
    parser.add_argument("-n" , "--number"       , required=True ,**argv,type=int, help="number of structures per file")
    parser.add_argument("-f" , "--folder"       , required=False,**argv,type=str, help="output folder (default: splitted)", default='splitted')
    parser.add_argument("-o" , "--output"       , required=True ,**argv,type=str, help="output prefix")
    parser.add_argument("-s" , "--suffix"       , required=False,**argv,type=str, help="suffix (default: .xyz)", default='.xyz')
    return parser

#---------------------------------------#
def count_lines(file_path):
    """
    Count the number of lines in a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The number of lines in the file.

    Raises:
        CalledProcessError: If the subprocess command fails.
    """
    # Use wc -l to count the number of lines in the file
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    line_count = int(result.stdout.split()[0])
    return line_count

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
            print(f"\tWarning: {split_file} does not have lines proportional to {lines_per_snapshot}.")
        n_files += 1

    print()
    if n_files != num_splits:
        print(f"\tError: Number of split files ({n_files}) does not match the expected number ({num_splits}).")
    else:
        print(f"\tSuccess: Number of split files matches the expected number.")

    # Count the lines in the original file
    original_lines = count_lines(input_file)

    if total_lines != original_lines:
        print(f"\tError: Total lines in split files ({total_lines}) do not match original file ({original_lines}).")
    else:
        print(f"\tSuccess: Total lines in split files match the original file.")

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    # Count the number of lines in the input file
    total_lines = count_lines(args.input)
    print(f'\n\tTotal lines in the file: {total_lines}')

    # Compute the number of lines per split file
    lines_per_snapshot = args.atoms + 2
    print(f'\tLines per snapshot: {lines_per_snapshot}')
    lines_per_file = args.number * lines_per_snapshot
    num_splits = total_lines // lines_per_file
    if total_lines % lines_per_file != 0:
        num_splits += 1

    print(f'\tNumber of splits: {num_splits}')
    print(f'\tLines per split file: {lines_per_file}')

    if args.folder is not None and args.folder not in ['', ' ']:
        os.makedirs(args.folder, exist_ok=True)
        args.output = os.path.join(args.folder, args.output)

    # Construct the split command
    split_command = [
        'split',
        '-l', str(lines_per_file),
        '-d', f'--additional-suffix={args.suffix}',
        args.input,
        args.output
    ]

    # Execute the split command
    subprocess.run(split_command, check=True)
    print(f'\n\tFile has been split into {num_splits} parts.')

    # Verify the split files
    verify_split(args.input, args.output, num_splits, lines_per_snapshot)

if __name__ == "__main__":
    main()
