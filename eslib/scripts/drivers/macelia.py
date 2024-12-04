#!/usr/bin/env python
# from mace.calculators import MACECalculator
import torch
from ase.calculators.socketio import SocketClient
from ase.io import read
from mace.calculators import MACECalculator, MACEliaCalculator

from eslib.drivers.socketextras import SocketClientExtras
from eslib.formatting import esfmt
from eslib.functions import suppress_output
from eslib.input import slist, str2bool
from warnings import warn

#---------------------------------------#
# Description of the script's purpose
description = "MACE socket driver."

choices = ["foundation_mp","mp","mace_mp",\
           "foundation_anicc","anicc","mace_anicc",\
           "foundation_off","off","mace_off",\
            "eslib","MACE", "MACElia"  ]

CHECK_QS = False

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-s" , "--structure"          , **argv, required=True , type=str     , help="file with the atomic structure")
    parser.add_argument("-f" , "--format"             , **argv, required=False, type=str     , help="file format of the atomic structure (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--model"              , **argv, required=False, type=str     , help="file with the MACE model (default: %(default)s)", default=None)
    parser.add_argument("-mt", "--model_type"         , **argv, required=True , type=str     , help="MACE model data type", choices=choices)
    parser.add_argument("-p" , "--port"               , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"            , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"               , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    parser.add_argument("-d" , "--device"             , **argv, required=False, type=str     , help="device (default: %(default)s)", choices=['cpu','gpu','cuda'], default='cuda')
    parser.add_argument("-dt", "--dtype"              , **argv, required=False, type=str     , help="dtype (default: %(default)s)", choices=['float64','float32'], default='float64')
    parser.add_argument("-sc", "--socket_client"      , **argv, required=False, type=str     , help="socket client (default: %(default)s)", choices=['eslib','ase'], default='eslib')
    parser.add_argument("-sp", "--suppress_properties", **argv, required=False, type=slist   , help="list of the properties to suppress (default: %(default)s)", default=None)
    parser.add_argument("-us" , "--use_stress"        , **argv, required=False, type=str2bool, help="use stress (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tCuda available: ",torch.cuda.is_available())
    if torch.cuda.is_available() and args.device != "cuda":
        warn(f"CUDA is available but you specified {args.device}. Let's move to CUDA.")
        args.device = "cuda"        

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
        "default_dtype" : args.dtype
    }
    args.model_type = str(args.model_type)# .lower()

    if args.model_type.lower() in ["foundation_mp","mp","mace_mp"]:
        print("\tLoading the MACECalculator with a pretrained model based on the Materials Project ... ", end="")
        from mace.calculators import mace_mp
        with suppress_output():
            calculator = mace_mp(model=args.model,**kwargv)

    elif args.model_type.lower() in ["foundation_anicc","anicc","mace_anicc"]:
        print("\tLoading the MACECalculator with a pretrained model based on the ANI (H, C, N, O) ... ", end="")
        from mace.calculators import mace_anicc
        with suppress_output():
            calculator = mace_anicc(device=args.device)

    elif args.model_type.lower() in ["foundation_off","off","mace_off"]:
        print("\tLoading the MACECalculator with a pretrained model based on the MACE-OFF23 models ... ", end="")
        from mace.calculators import mace_off
        with suppress_output():
            calculator = mace_off(model=args.model,**kwargv)

    elif args.model_type == "eslib":
        from eslib.classes.models.mace_model import MACEModel
        print("\tLoading eslib MACE model from file '{:s}' ... ".format(args.model), end="")
        calculator = MACEModel.from_pickle(args.model)
        try:
            calculator.to(device=args.device,dtype=args.dtype)
        except Exception as e:
            warn(f"Failed moving calculator to {args.device} or/and {args.dtype}.\n{e}")
        # try:
        #     calculator.to(dtype=args.dtype)
        # except Exception as e:
        #     warn(f"Failed moving calculator to {args.dtype}\n{e}")
        
    elif args.model_type == "MACE":
        print("\tLoading a MACECalculator based on the model that you provided ... ", end="")
        calculator = MACECalculator(model_paths=args.model,**kwargv)         
    elif args.model_type == "MACElia":
        print("\tLoading a MACEliaCalculator based on the model that you provided ... ", end="")
        calculator = MACEliaCalculator(model_paths=args.model,\
                                    model_type=args.model_type,\
                                    **kwargv)         
    print("done")

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
    socket_client = SocketClientExtras if args.socket_client == 'eslib' else SocketClient
    print("\tPreparing the socket communication ... ", end="")
    client = socket_client(host=args.address,\
                          port=args.port,\
                          unixsocket=args.address if args.unix else None)
    print("done")

    #------------------#
    print("\n\tRunning ... ", end="")
    client.run(atoms,use_stress=False)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()