#!/usr/bin/env python
import numpy as np
from ase import Atoms

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
from eslib.input import flist

#---------------------------------------#
# Description of the script's purpose
description = "Get the vertices of a unit cell."

vertices = np.array([
    [0, 0, 0],  # P0
    [1, 0, 0],  # P1
    [1, 1, 0],  # P2
    [0, 1, 0],  # P3
    [0, 0, 1],  # P4
    [1, 0, 1],  # P5
    [1, 1, 1],  # P6
    [0, 1, 1]   # P7
])


def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str  , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str  , help="output file (default: %(default)s)", default='vertices.txt')
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    cell = np.asarray(structure.get_cell()).T
    
    cartesian = (cell @ vertices.T).T
    
    assert np.allclose(cell[:,0],cartesian[1,:]), "error"
    assert np.allclose(cell[:,1],cartesian[3,:]), "error"
    assert np.allclose(cell[:,2],cartesian[4,:]), "error"
    
    
    print("\n\tVertices of the unit cell:")
    for n, v in enumerate(cartesian):
        print("\t - P{:d}: {}".format(n,list(v)))
    
    print("\n\tWriting vertices to file '{:s}' ... ".format(args.output), end="")
    with open(args.output, "w") as f:
        for n, v in enumerate(cartesian):
            f.write("P{:d} = ({:.6f}, {:.6f}, {:.6f})\n".format(n, v[0], v[1], v[2]))
    print("done")
    
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
