#!/usr/bin/env python3
import numpy as np

from eslib.formatting import esfmt
from eslib.input import str2bool
from eslib.physics import FWHM2sigma, sigma2FWHM
from eslib.tools import convert

#---------------------------------------#
description = "Full Width at Half Maximum (FWHM) to sigma."
documentation = "https://en.wikipedia.org/wiki/Full_width_at_half_maximum"

#---------------------------------------#
def prepare_parser(description):
    """set up the script input parameters"""
    import argparse
    argv = {"metavar" : "\b",}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i" , "--input"      , **argv, required=True , type=float   , help="input value")
    parser.add_argument("-iu", "--input_unit" , **argv, required=False, type=str     , help="input value unit (default: %(default)s)", default="femtosecond")
    parser.add_argument("-ou", "--output_unit", **argv, required=False, type=str     , help="output value unit(default: %(default)s)", default="femtosecond")
    parser.add_argument("-r" , "--revert"     , **argv, required=False, type=str2bool, help="convert sigma to FWHM (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description,documentation)
def main(args):
    
    period = convert(args.input,"time",args.input_unit,args.output_unit)
    if args.revert:
        print("\tsigma: {:.2f} {:s}".format(args.input,args.input_unit))
        period = sigma2FWHM(period)
        print("\t FWHM: {:.2f} {:s}".format(period,args.output_unit))
    else:
        print("\t FWHM: {:.2f} {:s}".format(args.input,args.input_unit))
        period = FWHM2sigma(period)
        print("\tsigma: {:.2f} {:s}".format(period,args.output_unit))
        
    return 0 

#---------------------------------------#
if __name__ == "__main__":
    main()