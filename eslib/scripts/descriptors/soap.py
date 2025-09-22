#!/usr/bin/env python
import numpy as np
from ase import Atoms
# from concurrent.futures import ProcessPoolExecutor, as_completed
from featomic import SoapPowerSpectrum
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Compute the SOAP descriptors for a bunch of atomic structures."

#---------------------------------------#
def prepare_parser(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"       , type=str, required=True , **argv, help="input file [au]")
    parser.add_argument("-if" , "--input_format", type=str, required=False, **argv, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-o"  , "--output"      , type=str, required=False, **argv, help="output file with SOAP descriptors (default: %(default)s)", default='soap.npy')
    # parser.add_argument("-j" , "--jobs"         , **argv, required=False, type=int  , help="number of parallel processes (default: %(default)s)", default=2)
    return parser

#---------------------------------------#
def process_structure(frame:Atoms,hypers:dict,calculator: SoapPowerSpectrum=None)->np.ndarray:
    if calculator is not None:
        calculator = SoapPowerSpectrum(**hypers)
    descriptor = calculator.compute(frame)
    descriptor = descriptor.keys_to_samples("center_type")
    descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    return descriptor.block().values.mean(axis=0)

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    structures = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tn. of structures: ",len(structures))
    
    print("\n\tPreparing SOAP object ... ",end="")
    
    # initialize SOAP
    HYPER_PARAMETERS = {
        "cutoff": {
            "radius": 5.0,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 4,
            "radial": {"type": "Gto", "max_radial": 6},
        },
    }

    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    print("done")
    
    # #-------------------#
    # if args.jobs > 1:
    #     print(f"\tProcessing {len(structures)} structures in parallel with {args.jobs} workers ...",end="")
    #     with ProcessPoolExecutor(max_workers=args.jobs) as executor:
    #         # ii = list(executor.map(process_structure, structures))
            
    #         inputs = [(atoms,HYPER_PARAMETERS) for atoms in structures]
    #         descriptors = list(executor.map(process_structure, inputs))
    #         # N = len(structures)
    #         # futures = {executor.submit(process_structure, frame, HYPER_PARAMETERS) for frame in structures}
    #         # descriptors = [None]*N
    #         # for future in as_completed(futures):
    #         #     n, des = future.result()
    #         #     descriptors[n] = des
    # else:
    print(f"\tProcessing {len(structures)} structures sequentially ...",end="")
    descriptors = [process_structure(atoms,HYPER_PARAMETERS,calculator) for atoms in  structures]
    descriptors = np.asarray(descriptors)
    print("done")
    print(f"\n\tSOAP features shape: {descriptors.shape}")

    #-------------------#
    print("\tSaving SOAP descriptors to file '{:s}' ... ".format(args.output),end="")
    if str(args.output).endswith("npy"):
        np.save(args.output, descriptors)
    elif str(args.output).endswith("txt"):
        np.savetxt(args.output, descriptors)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()
