#!/usr/bin/env python
import numpy as np
from ase.io.cube import read_cube_data
from sympy import E
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
    # read Hartree potential from a cube file
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,index=0)[0]
    cube = CubeData.from_file(args.input)
    print("done")
    assert np.allclose(atoms.cell.T,cube.vectors), "Cell and cube vectors differ."
    
    #------------------#
    # summary of the Hartree potential
    s = cube.summary(title="Hartee potential",prefix="\t"); print(s)
    
    #------------------#
    # compute the Electric field (gradient) using spatial Fourier transform
    print("\tComputing the electric field ... ",end="")
    Efield = cube.gradient()
    print("done")
    # summary of the Electric field
    s = Efield.summary(title="Electric field",prefix="\t"); print(s)       
    
    #------------------#
    # integrate the electric field to get the polarization
    print("\tComputing polarization ... ",end="")
    polarization = Efield.integrate()
    print("done")
    print("\tpolarization: ",polarization.tolist())
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()