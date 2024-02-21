#!/usr/bin/env python
from eslib.tools import convert

#---------------------------------------#

# Description of the script's purpose
description = "Convert a physical quantity from one measure unit to another."
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"


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
def correct(unit):
    if unit == "ang":
        return "angstrom"
    elif unit == "au":
        return "atomic_unit"
    else:
        return unit
    
#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-v", "--value", type=float, **argv, help="value")
    parser.add_argument("-f", "--family", type=str, **argv, help="family")
    parser.add_argument("-iu", "--in_unit", type=str, **argv, help="input unit")
    parser.add_argument("-ou", "--out_unit", type=str, **argv, help="output unit")
    return parser.parse_args()

#---------------------------------------#
def main():

    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #---------------------------------------#
    if args.family is None:
        raise ValueError("'family' can not be None")
    
    args.in_unit  = correct(args.in_unit)
    args.out_unit = correct(args.out_unit)
    factor = convert(1,args.family,args.in_unit,args.out_unit)

    print("\n\t{:>10s}: ".format("in-value"),args.value)
    print("\t{:>10s}: {:<s}".format("in-unit",args.in_unit))
    print("\t{:>10s}: {:<s}".format("out-unit",args.out_unit))
    print("\t{:>10s}: ".format("factor"),factor)
    print("\t{:>10s}: ".format("out-value"),factor*args.value)

    print("\n\t{:f} {:s} = {:f} {:s}".format(args.value,args.in_unit,factor*args.value,args.out_unit))

    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()