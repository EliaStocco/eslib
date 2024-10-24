#!/usr/bin/env python
from ase.calculators.socketio import SocketClient
from ase.io import read

from eslib.classes.potentials.LJwall_calculator import LennardJonesWall
from eslib.formatting import esfmt
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "LennardJonesWall socket driver."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-s" , "--structure"   , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-f" , "--format"      , **argv, required=False, type=str     , help="file format of the atomic structure (default: %(default)s)" , default=None)
    parser.add_argument("-i" , "--instructions", **argv, required=True , type=str     , help="file with the instuctions for the calulator")
    parser.add_argument("-l" , "--logger"      , **argv, required=False, type=str     , help="logging file (default: %(default)s)", default=None)
    parser.add_argument("-p" , "--port"        , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"     , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.structure), end="")
    atoms = read(args.structure,format=args.format,index=0)
    print("done")

    #------------------#
    # I don't know if I actually need this line
    atoms.info = {}

    
    print("\tLoading the LennardJonesWall calculator ... ", end="")
    calculator = LennardJonesWall(instructions=args.instructions,log_file=args.logger)         
    print("done")

    atoms.calc = calculator

    #------------------#
    print("\tPreparing the socket communication ... ", end="")
    client = SocketClient(host=args.address,\
                          port=args.port,\
                          unixsocket=args.address if args.unix else None)
    print("done")

    #------------------#
    print("\n\tRunning ... ", end="")
    client.run(atoms)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()