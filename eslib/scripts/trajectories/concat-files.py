#!/usr/bin/env python
import subprocess
from eslib.io_tools import pattern2sorted_files
from eslib.formatting import esfmt
import numpy as np

#---------------------------------------#
# Description of the script's purpose
description = "Concatenate files."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input" , **argv, required=True , type=str, help="input files")
    parser.add_argument("-o", "--output", **argv, required=False, type=str, help="output file")
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    files = pattern2sorted_files(args.input)
    print("done")
    print("\tn. of files: ",len(files))
    
    #------------------#
    # Construct the 'cat' command
    cat_command = ['cat'] + files

    # cmd = ' '.join(cat_command)
    # print(f'\n\tRunning the following command:\n\t{cmd} > {args.output}')

    # Execute the 'cat' command
    try:
        with open(args.output, 'wb') as output_file:
            subprocess.run(cat_command, stdout=output_file, check=True)
        print(f"\n\tFiles concatenated into '{args.output}' successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n\tError: Command failed with {e}")
    except Exception as e:
        print(f"\n\tUnexpected error: {e}")
        
    # test = np.loadtxt(args.output)
    # print(test.shape)

#---------------------------------------#
if __name__ == "__main__":
    main()