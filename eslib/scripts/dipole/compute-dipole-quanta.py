#!/usr/bin/env python
import argparse
import numpy as np
from eslib.tools import cart2frac
from ase.io import read
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dipole quanta."

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv,type=str, help="input file")
    parser.add_argument("-if", "--input_format" , **argv,type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-k"  , "--keyword"     , **argv,type=str, help="keyword (default: 'dipole')" , default="dipole")
    parser.add_argument("-o" , "--output"       , **argv,type=str, help="txt output file (default: 'quanta.txt')", default="quanta.txt")
    parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: '%%24.18e')", default='%24.18e')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=":")
    print("done")

    # #---------------------------------------#
    # # dipole
    # print("\tExtracting '{:s}' from the trajectory ... ".format(args.keyword), end="")
    # dipole = atoms.call(lambda e:e.info[args.keyword])
    # print("done")

    # #---------------------------------------#
    # # lattice vectors
    # print("\tExtracting the lattice vectors from the trajectory ... ", end="")
    # lattices = atoms.call(lambda e:e.get_cell())
    # print("done")

    #---------------------------------------#
    # pbc
    # # pbc = atoms.call(lambda e:e.pbc)
    # if not np.all([ np.all(a.get_pbc()) for a in atoms ]):
    #     raise ValueError("The system is not periodic.")

    #---------------------------------------#
    # quanta
    print("\tComputing the dipole quanta ... ", end="")
    N = len(atoms)
    quanta = np.zeros((N,3))
    # for n in range(N):
    #     cell = lattices[n].T
    #     R = cart2lattice(lattices[n])
    #     lenght = np.linalg.norm(cell,axis=0)
    #     quanta[n,:] = R @ dipole[n] / lenght
    for n in range(N):
        # atoms[n].set_calculator(None)
        # cell = np.asarray(atoms[n].cell.array).T
        # lenght[n,:] = np.linalg.norm(cell,axis=0)
        # R = cart2lattice(cell)
        # dipole = R @ atoms[n].info[args.name]
        # phases[n,:] = dipole / lenght[n,:]
        quanta[n,:] = cart2frac(cell=atoms[n].get_cell(),v=atoms[n].info[args.keyword])
    print("done")

    #---------------------------------------#
    # output
    print("\n\tWriting dipole quanta to file '{:s}' ... ".format(args.output), end="")
    try:
        np.savetxt(args.output,quanta,fmt=args.output_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

if __name__ == "__main__":
    main()

