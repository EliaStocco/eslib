#!/usr/bin/env python
import json

import numpy as np
# Some problems with librascal
# Import `rascal` after `pandas`
from featomic import SphericalExpansion as SOAP
from featomic import SoapPowerSpectrum
from tqdm.auto import tqdm

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.functions import add_default
from eslib.show import show_dict

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
    parser.add_argument("-sh" , "--soap_hyper"  , type=str, required=False, **argv, help="JSON file with the SOAP hyperparameters (default: %(default)s)", default=None)
    parser.add_argument("-o"  , "--output"      , type=str, required=False, **argv, help="output file with SOAP descriptors (default: %(default)s)", default='soap.npy')
    return parser

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    frames = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")

    # print("\n\tConverting the atomic structures positions and cells from atomic unit to angstrom ... ",end="")
    # factor = convert(1,"length","atomic_unit","angstrom")
    # for n in range(len(frames)):
    #     frames[n].positions *= factor
    #     if np.any(frames[n].pbc):
    #         cell = factor * frames[n].get_cell()
    #         frames[n].set_cell(cell)
    # print("done")

    available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
    available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

    print('\tNumber of frames: ', len(frames))
    print('\tNumber of atoms/frame: ', len(frames[0]))
    print('\tAvailable structure properties: ', available_structure_properties)
    print('\tAvailable atom-level properties: ', available_atom_level_properties)

    if args.soap_hyper is not None :
        print("\n\tReading the SOAP hyperparameters from file '{:s}' ... ".format(args.soap_hyper),end="")
        with open(args.soap_hyper, 'r') as file:
            # Load the JSON data from the file
            user_soap_hyper = json.load(file)
        print("done")
    else:
        user_soap_hyper = None

    print("\n\tUsing the following SOAP hyperparameters:")
    SOAP_HYPERS = {
        "cutoff": {
            "radius": 6.0,
            "smoothing": {
                "type": "ShiftedCosine",
                "width": 0.1
            }
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 6,
            "radial": {
                "type": "Gto",
                "max_radial": 5
            },
            "spline_accuracy": 1e-06
        }
    }

    SOAP_HYPERS = add_default(user_soap_hyper,SOAP_HYPERS)
    # SOAP_HYPERS["interaction_cutoff"] = args.cutoff_radius
    show_dict(SOAP_HYPERS,string="\t\t")

    #
    print("\n\tPreparing SOAP object ... ",end="")
    numbers = list(sorted(set([int(n) for frame in frames for n in frame.numbers])))

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
    
    descriptors = [None]*len(frames)
    for n,frame in enumerate(frames):
        descriptor = calculator.compute(frame)
        descriptor = descriptor.keys_to_samples("center_type")
        descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        descriptors[n] = descriptor.block().values.mean(axis=0)

    # soap = SOAP(**SOAP_HYPERS)
    #     # global_species=numbers,
    #     # expansion_by_species_method='user defined',
    #     # **SOAP_HYPERS
    # # )
    print("done")

    X = None
    print("\tComputing SOAP features ... ")
    for i, frame in enumerate(tqdm(frames)):
        # normalize cell for librascal input
        if np.linalg.norm(frame.cell) < 1e-16:
            extend = 1.5 * (np.max(frame.positions.flatten()) - np.min(frame.positions.flatten()))
            frame.cell = [extend, extend, extend]
            frame.pbc = True
        frame.wrap(eps=1e-16)

        x = soap.compute(frame).get_features(soap).mean(axis=0) # here it takes mean over atoms in the frame
        if X is None:
            X = np.zeros((len(frames), x.shape[-1]))
        X[i] = x

    print(f"\n\tSOAP features shape: {X.shape}")

    print("\tSaving SOAP descriptors to file '{:s}' ... ".format(args.output),end="")
    if str(args.output).endswith("npy"):
        np.save(args.output, X)
    elif str(args.output).endswith("txt"):
        np.savetxt(args.output, X)
    print("done")


#---------------------------------------#
if __name__ == "__main__":
    main()
