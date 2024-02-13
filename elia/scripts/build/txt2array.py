#!/usr/bin/env python
import argparse
import numpy as np
import re
from elia.input import size_type

#---------------------------------------#
# Description of the script's purpose
description = "Extract an array from a txt file using a keyword."
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"

#---------------------------------------#
def count_occurrences(log_content, keyword):
    pattern = re.compile(f'{keyword}:', re.MULTILINE)
    return len(pattern.findall(log_content))

def extract_matrix(file, keyword, shape:tuple):
    with open(file, 'r') as f:
        log_content = f.read()

    num_occurrences = count_occurrences(log_content, keyword)
    arrays = np.zeros((num_occurrences,*shape))

    N = shape[0]
    k = -1
    n = 0
    # Open the file in read mode
    with open(file, 'r') as f:
        # Initialize variables
        storing_lines = False
        accumulated_lines = [""]*N

        # Iterate over each line in the file
        for line in f:
            # Check if the keyword is in the line
            if keyword+":" in line:
                # Set the flag to start storing lines
                storing_lines = True
                #accumulated_lines = []
            
            # If storing lines is active, accumulate the line
            if storing_lines:
                if k >= 0 :
                    accumulated_lines[k] = line.strip()
                k += 1

                # Check if N lines have been accumulated
                if k == N:
                    # Perform manipulations on the accumulated lines
                    for i in range(N):
                        # for j in range(shape[1]):
                        row = accumulated_lines[i]
                        ready4float = row.replace("[","").replace("]","").replace(",","").split()
                        #result_list = ast.literal_eval("[{:s}]".format(ready4float))
                        #ready4float = ready4float[[ a != "" for a in ready4float]] 
                        ready4numpy = [ float(a) for a in ready4float ]
                        arrays[n,i,:] = np.asarray(ready4numpy)
                    n += 1
                    
                    storing_lines = False
                    k = -1
            # else:
            #     # Process other lines as needed
            #     print(f"Processing line: {line.strip()}")
    return arrays



def extract_arrays(file, keyword, shape:tuple=None):
    with open(file, 'r') as f:
        log_content = f.read()

    num_occurrences = count_occurrences(log_content, keyword)
    arrays = [None] * num_occurrences

    # Use regular expression to find arrays
    pattern = re.compile(f'{keyword}:\s*\[([\s\S]*?)\]', re.MULTILINE)
    matches = pattern.finditer(log_content)

    for i, match in enumerate(matches):
        # Split the matched string into lines and convert to a 1D array
        lines = match.group(1).strip().split('\n')
        arrays[i] = list(map(float, lines[0].split()))
        if shape is not None:
            try:
                arrays[i] = np.asarray(arrays[i]).reshape(shape)
            except:
                raise ValueError("the array can not converted into array of shape ",shape)

    return np.asarray(arrays)

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
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"       , **argv,type=str      , help="input file")
    parser.add_argument("-k"  , "--keyword"     , **argv,type=str      , help="keyword")
    parser.add_argument("-s"  , "--shape"       , **argv,type=size_type, help="shape of the array (default: 'None')", default=None)
    parser.add_argument("-o"  , "--output"      , **argv,type=str      , help="txt output file (default: '[keyword].txt')", default=None)
    return parser.parse_args()

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
    # read atomic structures
    print("\tExtracting '{:s}' from file '{:s}' ... ".format(args.keyword,args.input), end="")
    shape = tuple(args.shape) if args.shape is not None else None
    dimension = len(shape) if shape is not None else 1
    if dimension == 1:
        array = extract_arrays(args.input,args.keyword,shape)
    elif dimension == 2:
        array = extract_matrix(args.input,args.keyword,shape)
    print("done")

    #------------------#
    # show
    print("\tExtracted array has shape: ",array.shape)

    #------------------#
    # output shape
    array = array.reshape((array.shape[0],-1))
    print("\tThe array has been reshaped to: ",array.shape)

    #------------------#
    # output
    if args.output is None:
        args.output = "{:s}.txt".format(args.keyword)

    if args.output.lower() != "none" and len(args.output) > 0 :
        print("\n\tWriting array to file '{:s}' ... ".format(args.output), end="")
        try:
            np.savetxt(args.output,array)
            print("done")
        except Exception as e:
            print("\n\tError: {:s}".format(e))

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()

