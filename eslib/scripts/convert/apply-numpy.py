#!/usr/bin/env python
import numpy as np
from eslib.tools import string2function
#---------------------------------------#
# Description of the script's purpose
description = "Apply a function to an array read from a txt file, and save the result to another txt file."
error = "***Error***"
closure = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , required=True,**argv,type=str, help="txt input file")
    parser.add_argument("-f" , "--function"     , required=True,**argv,type=str, help="source code of the function to be applied")
    parser.add_argument("-o" , "--output"       , required=True,**argv,type=str, help="txt output file")
    parser.add_argument("-of", "--output_format", required=False,**argv,type=str, help="txt output format for np.savetxt (default: '%%24.18f')", default='%24.18f')
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

    #---------------------------------------#
    print("\tConverting string into function ... ", end="")
    function = string2function(args.function)
    print("done")

    #---------------------------------------#
    print("\tReading array from file '{:s}' ... ".format(args.input), end="")
    inarray = np.loadtxt(args.input)
    print("done")

    print("\tinput array shape: ",inarray.shape)

    #---------------------------------------#
    print("\tApplying function to the array ... ", end="")
    outarray = function(inarray)
    print("done")

    print("\toutput array shape: ",outarray.shape)
    
    #---------------------------------------#
    if args.output is None:
        print("\t{:s}: no output file provided.\nSpecify it with -o,--output")
    else:
        print("\tSave output array to file '{:s}' ... ".format(args.output), end="")
        np.savetxt(args.output,outarray,fmt=args.output_format)
        print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()

