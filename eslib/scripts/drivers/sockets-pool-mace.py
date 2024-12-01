#!/usr/bin/env python
# from mace.calculators import MACECalculator
import torch
from ase.calculators.socketio import SocketClient
from ase.io import read
from eslib.drivers.socketextras import SocketClientExtras
from eslib.formatting import esfmt
from eslib.functions import suppress_output
from eslib.input import str2bool, slist, ilist, blist
from warnings import warn

from eslib.classes.calculators.pool import SocketsPoolMACE

#---------------------------------------#
# Description of the script's purpose
description = "Connect a MACE models to many sockets."

CHECK_QS = False

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"              , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-if", "--input_format"       , **argv, required=False, type=str     , help="file format of the atomic structure (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--model"              , **argv, required=False, type=str     , help="file with the MACE model (default: %(default)s)", default=None)
    parser.add_argument("-p" , "--port"               , **argv, required=False, type=ilist   , help="TCP/IP port numbers. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"            , **argv, required=True , type=slist   , help="Host names (for INET sockets) or names of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"               , **argv, required=True , type=blist   , help="Use a UNIX domain socket")
    parser.add_argument("-d" , "--device"             , **argv, required=False, type=str     , help="device (default: %(default)s)", choices=['cpu','gpu','cuda'], default='cuda')
    parser.add_argument("-dt", "--dtype"              , **argv, required=False, type=str     , help="dtype (default: %(default)s)", choices=['float64','float32'], default='float64')
    parser.add_argument("-sc", "--socket_client"      , **argv, required=False, type=str     , help="socket client (default: %(default)s)", choices=['eslib','ase'], default='eslib')
    parser.add_argument("-sp", "--suppress_properties", **argv, required=False, type=slist   , help="list of the properties to suppress (default: %(default)s)", default=None)
    parser.add_argument("-us", "--use_stress"         , **argv, required=False, type=str2bool, help="use stress (default: %(default)s)", default=False)
    parser.add_argument("-l" , "--logger"             , **argv, required=False, type=str     , help="logging file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tCuda available: ",torch.cuda.is_available())

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format=args.input_format,index=0)
    print("done")

    #------------------#
    # I don't know if I actually need this line
    atoms.info = {}

    #------------------#
    kwargv = {
        "device" : args.device,
        "default_dtype" : args.dtype
    }
   
    #------------------#
    if str(args.model).endswith(("pickle")):
        from eslib.classes.models.mace_model import MACEModel
        print("\tLoading eslib MACE model from file '{:s}' ... ".format(args.model), end="")
        calculator = MACEModel.from_pickle(args.model)
        try:
            calculator.to(device=args.device)
        except:
            warn(f"Failed moving calculator to {args.device}.")
        try:
            calculator.to(dtype=args.dtype)
        except:
            warn(f"Failed moving calculator to {args.dtype}.")
        
    else:
        raise ValueError("not implemented yet")
        from mace.calculators import MACECalculator
        print("\tLoading a MACECalculator based on the model that you provided ... ", end="")
        calculator = MACECalculator(model_paths=args.model,**kwargv)         
    print("done")

    #------------------#
    if calculator is None:
        raise Exception("Calculator is None. Some errors may have occurred. Please check the input arguments.")

    try:
        calculator.summary()
    except:
        pass

    #------------------#
    if args.suppress_properties is not None:
        print(f"\n\tSuppressing properties:")
        for prop in args.suppress_properties:
            print(f"\t - '{prop}' ... ", end="")
            calculator.implemented_properties.pop(prop)
            print("done")

        try:
            calculator.summary()
        except:
            pass
        
    #------------------#
    if CHECK_QS:
        if hasattr(calculator,"charges_key"):
            if calculator.charges_key not in atoms.arrays:
                raise ValueError("The atomic structures do not have the key '{:s}'".format(calculator.charges_key))

    atoms.calc = calculator # atoms.set_calculator(calculator)
    
    
    #------------------#
    if args.port is None:
        ports = [ None for _ in zip(args.address) ]
    else:
        ports = args.port
     
    unixsockets = [ a if u else None for a,u in zip(args.address,args.unix) ]
    log = args.logger
    
    print("\tPreparing the socket communication ... ", end="")
    client = SocketsPoolMACE(ports,unixsockets,args.socket_client,log,calculator)
    print("done")

    #------------------#
    print("\n\tRunning ... ", end="")
    client.run(atoms,use_stress=args.use_stress)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()