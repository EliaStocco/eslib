#!/usr/bin/env python3
import numpy as np
from eslib.tools import convert
from eslib.formatting import esfmt

#---------------------------------------#
description = "Compute the period given its angular frequency."

#---------------------------------------#
def prepare_parser(description):
    """set up the script input parameters"""
    import argparse
    argv = {"metavar" : "\b",}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-w" , "--omega"      , required=True , type=float, help="angular frequency", default="input.xml")
    parser.add_argument("-wu", "--omega_unit" , required=False, type=str  , help="angular frequency unit (default: %(default)s)", default="THz")
    parser.add_argument("-tu", "--period_unit", required=False, type=str  , help="period unit (default: %(default)s)", default="femtosecond")
    return parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):
    
    omega = convert(args.omega,"frequency",args.omega_unit,"atomic_unit")
    T = 2*np.pi/omega
    T = convert(T,"time","atomic_unit",args.period_unit)

    print("\tang. freq.: {:.2e} {:s}".format(args.omega,args.omega_unit))
    print("\t    period: {:.2e} {:s}".format(T,args.period_unit))

    return 0 

#---------------------------------------#
if __name__ == "__main__":
    main()