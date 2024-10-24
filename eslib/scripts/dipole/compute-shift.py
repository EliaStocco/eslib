#!/usr/bin/env python
import numpy as np
from ase.io import read

from eslib.formatting import esfmt
from eslib.input import flist
from eslib.tools import cart2frac

#---------------------------------------#
description = "Convert the shift-vector (dipole) from cartesian to fractional coordinates (quanta)."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, type=str  , required=True , help="input file [au]")
    parser.add_argument("-if", "--input_format", **argv, type=str  , required=False, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--shift"       , **argv, type=flist, required=True , help="additional arrays to be added to the output file (default: %(default)s)", default=None,)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if args.shift is not None:
        if len(args.shift) != 3 :
            raise ValueError("You should provide 3 integer numbers as shift vectors")
    
    #------------------#
    print("\tReading (the first only) atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format)
    print("done")

    #------------------#
    shift = cart2frac(cell=atoms.get_cell(),v=args.shift).flatten()
    print("\tConverted the shift from cartesian to lattice coordinates: ",np.round(shift,0))
    print("\tShift with all digits: ",shift)

#---------------------------------------#
if __name__ == "__main__":
    main()
