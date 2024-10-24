#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from phonopy.units import VaspToTHz

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import esfmt, warning
from eslib.functions import phonopy2atoms
from eslib.geometry import modular_norm
from eslib.input import size_type
from eslib.output import output_folder
from eslib.show import matrix2str
from eslib.tools import convert, is_sorted_ascending, w2_to_w

#---------------------------------------#
# Description of the script's purpose
description = "Compute the center of mass of a structure."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    #------------------#
    print("\n\tComputing center of mass ... ", end="")
    com = structure.get_center_of_mass()
    print("done")
    
    print("\tCenter of mass [ang]:",com)
        
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
