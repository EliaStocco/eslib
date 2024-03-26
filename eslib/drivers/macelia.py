#!/usr/bin/env python
# from mace.calculators import MACECalculator
from mace.calculators import MACEliaCalculator
from ase.io import read
from eslib.formatting import esfmt
from eslib.functions import suppress_output
from eslib.input import str2bool
from eslib.drivers.socketextras import SocketClientExtras
# from ase.calculators.socketio import SocketClient

#---------------------------------------#
# Description of the script's purpose
description = "MACE socket driver."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-s", "--structure" , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-f", "--format"    , **argv, required=False, type=str     , help="file format of the atomic structure (default: 'None')" , default=None)
    parser.add_argument("-m", "--model"     , **argv, required=False, type=str     , help="file with the MACE model")
    parser.add_argument("-t", "--model_type", **argv, required=True , type=str     , help="MACE model data type (default: None)", default=None)
    parser.add_argument("-p", "--port"      , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a", "--address"   , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u", "--unix"      , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: false)", default=False)
    parser.add_argument("-d", "--device"    , **argv, required=False, type=str     , help="device (default: 'cpu')", choices=['cpu','gpu','cuda'], default='cpu')
    return parser.parse_args()

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

    #------------------#
    kwargv = {
        "device" : args.device,
        "default_dtype" : 'float64'
    }
    args.model_type = str(args.model_type).lower()
    if args.model_type in ["foundation_mp","mp","mace_mp"]:
        print("\tLoading the MACECalculator with a pretrained model based on the Materials Project ... ", end="")
        from mace.calculators import mace_mp
        with suppress_output():
            calculator = mace_mp(model=args.model,**kwargv)
    elif args.model_type in ["foundation_anicc","anicc","mace_anicc"]:
        from mace.calculators import mace_anicc
        with suppress_output():
            calculator = mace_anicc(model=args.model,**kwargv)
    elif args.model_type in ["foundation_off","off","mace_off"]:
        from mace.calculators import mace_off
        with suppress_output():
            calculator = mace_off(model=args.model,**kwargv)
    else:
        print("\tCreating the MACECalculator ... ", end="")
        # try:
        #     calculator = MACECalculator(model_paths=args.model,\
        #                                 model_type=args.model_type,\
        #                                 **kwargv)
        # except:
        calculator = MACEliaCalculator(model_paths=args.model,\
                                    model_type=args.model_type,\
                                    **kwargv)         
    print("done")

    atoms.set_calculator(calculator)

    #------------------#
    print("\tPreparing the socket communication ... ", end="")
    client = SocketClientExtras(host=args.address,\
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