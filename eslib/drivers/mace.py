#!/usr/bin/env python
from mace.calculators.mace import MACECalculator
from ase.io import read
from ase.calculators.socketio import SocketClient
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "MACE socket driver."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-s", "--structure" , **argv, required=True , type=str, help="file with the atomic structure")
    parser.add_argument("-f", "--format"    , **argv, required=False, type=str, help="file format of the atomic structure (default: 'None')" , default=None)
    parser.add_argument("-m", "--model"     , **argv, required=True , type=str, help="file with the MACE model")
    parser.add_argument("-t", "--model_type", **argv, required=False, type=str, help="MACE model data type (default: None)", default=None)
    parser.add_argument("-p", "--port"      , **argv, required=True , type=str, help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a", "--address"   , **argv, required=True , type=str, help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u", "--unix"      , **argv, required=True , type=str, help="Use a UNIX domain socket.")
    parser.add_argument("-d", "--device"    , **argv, required=True , type=str, help="device (default: 'cpu')", choices=['cpu','gpu','cuda'], default='cpu')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.structure), end="")
    atoms = read(args.structure,format=args.format,index=0)
    print("done")

    #------------------#
    print("\tCreating the MACECalculator ... ", end="")
    calculator = MACECalculator(args.model, device=args.device, default_dtype='float64')
    print("done")

    atoms.set_calculator(calculator)

    #------------------#
    print("\tPreparing the socket communication ... ", end="")
    client = SocketClient(host=args.address, port=args.port)
    print("done")

    #------------------#
    print("\tRunning ... ", end="")
    client.run(atoms)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()