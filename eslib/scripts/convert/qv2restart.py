#!/usr/bin/env python
import ast
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
from ase import Atoms
from ase.io import write

from eslib.formatting import esfmt
from eslib.tools import convert

#---------------------------------------#
# Description of the script's purpose
description = "Insert the positions 'q' and the velocities 'v' to a RESTART file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-q" , "--positions"      , **argv,required=True , type=str, help="*.txt file with the positions")
    parser.add_argument("-v" , "--velocities"     , **argv,required=True , type=str, help="*.txt file with the velocities")
    parser.add_argument("-qu", "--positions_unit" , **argv,required=False, type=str, help="positions unit (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-vu", "--velocities_unit", **argv,required=False, type=str, help="velocities unit (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-r" , "--restart"        , **argv,required=True , type=str, help="template RESTART file")
    parser.add_argument("-o" , "--output"         , **argv,required=True , type=str, help="output file")
    return parser


#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()