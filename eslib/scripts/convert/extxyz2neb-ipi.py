#!/usr/bin/env python
from ase.calculators.socketio import SocketClient
from ase import Atoms, io
from eslib.classes.atomic_structures import AtomicStructures

from eslib.classes.potentials.LJwall_calculator import LennardJonesWall
from eslib.formatting import esfmt
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Torch-pme calculator."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file with the atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"     , **argv, required=True , type=str     , help="keyword for the charges (default: %(default)s)", default='Qs')
    parser.add_argument("-p" , "--port"        , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"     , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print(f'\t Number of atomic structures: {len(atoms)}')
    
    #------------------#
    print("\nWriting the atomic structures to file '{:s}' ... ".format(args.output), end="")
    for n,atoms in enumerate(structures):
        params = atoms.get_cell().cellpar()
        float_format = '%15.10e'
        fmt_header = "# CELL(abcABC): {:s}  {:s}  {:s}  {:s}  {:s}  {:s}  Bead: {:n} %s".format(*([float_format]*6,n))
        string = " positions{angstrom} cell{angstrom}"
        comment = fmt_header%(*params,string)
        io.write(args.output, atoms, format="xyz", append=True, comment=comment)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()