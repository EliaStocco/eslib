#!/usr/bin/env python3
import numpy as np

from eslib.formatting import esfmt
from eslib.tools import convert
from eslib.input import str2bool

#---------------------------------------#
description = "Compute the fluence of an electric field (plane wave + gaussian envelope)."

#---------------------------------------#
def prepare_parser(description):
    """set up the script input parameters"""
    import argparse
    argv = {"metavar" : "\b",}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v" , "--freq"       , **argv, required=True , type=float, help="frequency")
    parser.add_argument("-vu", "--freq_unit"  , **argv, required=False, type=str  , help="frequency unit (default: %(default)s)", default="THz")
    parser.add_argument("-s" , "--sigma"      , **argv, required=True , type=float, help="sigma")
    parser.add_argument("-su", "--sigma_unit" , **argv, required=False, type=str  , help="period unit (default: %(default)s)", default="femtosecond")
    parser.add_argument("-E" , "--Efield"     , **argv, required=False, type=float, help="Electric field value (default: %(default)s)", default=None)
    parser.add_argument("-Eu", "--Efield_unit", **argv, required=False, type=str  , help="Electric field unit (default: %(default)s)", default="V/ang")
    parser.add_argument("-F" , "--fluence"     , **argv, required=False, type=float, help="fluence (default: %(default)s)", default=None)
    parser.add_argument("-Fu", "--fluence_unit", **argv, required=False, type=str  , help="fluence unit (default: %(default)s)", default="mJ/cm2")
    parser.add_argument("-r" , "--revert"     , **argv, required=False, type=str2bool, help="fluence to electric field (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
epsilon_0 = 1/(4*np.pi)
c = convert(299792458,"velocity","m/s","atomic_unit")
    
#---------------------------------------#
def Efield2fluence(omega,sigma,Efield):
    """
    Calculate the fluence of an electric field with a plane wave and gaussian envelope.

    Parameters:
        - omega (float): angular frequency of the electric field [atomic units].
        - sigma (float): width of the Gaussian envelope [atomic units].
        - Efield (float): magnitude of the electric field [atomic units].

    Returns:
        float: the computed fluence [atomic units].
    """
    E2 = Efield**2
    S2 = sigma**2
    W2 = omega**2
    factor = np.sqrt(np.pi)/2. * (1+np.exp(-S2*W2))
    return epsilon_0*c*E2*sigma*factor

#---------------------------------------#
def fluence2Efield(omega,sigma,fluence):
    """
    Calculate the electric field magnitude of an electric field with a plane wave and gaussian envelope.

    Parameters:
        - omega (float): angular frequency of the electric field [atomic units].
        - sigma (float): width of the Gaussian envelope [atomic units].
        - fluence (float): value of the fluence [atomic units].

    Returns:
        float: the computed electric field magnitude [atomic units].
    """
    S2 = sigma**2
    W2 = omega**2
    factor = np.sqrt(np.pi)/2. * (1+np.exp(-S2*W2))
    E2 = fluence/(epsilon_0*c*sigma*factor)
    return np.sqrt(E2)

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):
    
    omega = convert(args.freq,"frequency",args.freq_unit,"atomic_unit")
    sigma = convert(args.sigma,"time",args.sigma_unit,"atomic_unit")
    
    if not args.revert:
        assert args.Efield is not None, "Electric field value must be provided when converting from fluence to electric field."
        Efield = convert(args.Efield,"electric-field",args.Efield_unit,"atomic_unit")
    
        F = Efield2fluence(omega,sigma,Efield)
        F = convert(F,"fluence","atomic_unit",args.fluence_unit)
    
        print("\tFluence: {:.2f} {:s}".format(F,args.fluence_unit))
    
    else:
        assert args.fluence is not None, "Fluence value must be provided when converting from electric field to fluence."
        F = convert(args.fluence,"fluence",args.fluence_unit,"atomic_unit")
    
        Efield = fluence2Efield(omega,sigma,F)
        Efield = convert(Efield,"electric-field","atomic_unit",args.Efield_unit)
    
        print("\tElectric field: {:.2f} {:s}".format(Efield,args.Efield_unit))
        
    return 0 

#---------------------------------------#
if __name__ == "__main__":
    main()