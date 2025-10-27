#!/usr/bin/env python
import numpy as np
from ase.io.cube import read_cube_data
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog
from eslib.tools import convert
from eslib.classes.cube import CubeData

#---------------------------------------#
description = "Compute the polarization given the cube file containing the Hartree potential."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str, help="input file")
    parser.add_argument("-if", "--input_format"    , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default='cube')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,index=0)[0]
    cube = CubeData.from_file(args.input)
    print("done")
    assert np.allclose(atoms.cell.T,cube.vectors), "Cell and cube vectors differ."
    
    #------------------#
    s = cube.summary(title="Hartee potential",prefix="\t"); print(s)
    
    #------------------#
    k = cube.kspace()
    r = k.rspace()
    assert r == cube, "codin error"
    
    
    #------------------#
    print("\tComputing the electric field ... ",end="")
    Efield = cube.gradient()
    print("done")
    Efield.summary(title="Electric field")       
    return
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()