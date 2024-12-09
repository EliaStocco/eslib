#!/usr/bin/env python
from ase.io import read
from ase.calculators.socketio import SocketClient
from eslib.drivers.socketextras import SocketClientExtras
from eslib.formatting import esfmt
from eslib.input import str2bool
from eslib.classes.calculators.fileIOcalculator import FileIOCalculator

#---------------------------------------#
description = "Run one FileIOCalculator."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str     , help="file format of the atomic structure (default: %(default)s)" , default=None)
    parser.add_argument("-p" , "--port"         , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"      , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"         , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    parser.add_argument("-sc", "--socket_client", **argv, required=False, type=str     , help="socket client (default: %(default)s)", choices=['eslib','ase'], default='ase')
    parser.add_argument("-ip", "--impl_prop"    , **argv, required=False, type=str     , help="implemented properties (default: %(default)s)", default=None)
    parser.add_argument("-f" , "--folder"       , **argv, required=True , type=str     , help="folder where the files will be written and read")
    parser.add_argument("-l" , "--log_file"     , **argv, required=False, type=str     , help="optional path for the log file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tAllocating the calculator ... ", end="")
    calculator = FileIOCalculator(folder=args.folder,log_file=args.log_file,impl_prop=args.impl_prop)
    print("done")
    print("\tImplemented properties:")
    max_key_length = max(len(k) for k in calculator.implemented_properties.keys())
    for k in calculator.implemented_properties.keys():
        print(f"\t - {k:<{max_key_length}}: {calculator.implemented_properties[k]}")
        
    #------------------#
    print("\n\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=0)
    atoms.info = {} # I don't know if I actually need this line
    print("done")

    atoms.calc = calculator

    #------------------#
    socket_client = SocketClientExtras if args.socket_client == 'eslib' else SocketClient
    print("\tPreparing the socket communication ... ", end="")
    client = socket_client(host=args.address,\
                          port=args.port,\
                          unixsocket=args.address if args.unix else None)
    print("done")

    #------------------#
    print("\n\tRunning ... \n")
    client.run(atoms,use_stress=False)
    print("\n\tDone")

#---------------------------------------#
if __name__ == "__main__":
    main()