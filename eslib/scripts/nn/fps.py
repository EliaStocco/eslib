#!/usr/bin/env python
from ase.io import read, write
import numpy as np
from skmatter.feature_selection import FPS
from eslib.input import str2bool
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Process atomic structures and select a diverse subset using the Farthest Point Sampling (FPS) algorithm."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"           , type=str     , required=True , **argv, help="input file [au]")
    parser.add_argument("-if" , "--input_format"    , type=str     , required=False, **argv, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-n"  , "--number"          , type=int     , required=True , **argv, help="number of desired structure")
    parser.add_argument("-s"  , "--sort"            , type=str2bool, required=False, **argv, help="whether to sort the indices (default: true)", default=True)
    parser.add_argument("-x"  , "--soap_descriptors", type=str     , required=True , **argv, help="file with the SOAP descriptors")
    parser.add_argument("-oi" , "--output_indices"  , type=str     , required=False, **argv, help="output file with indices of the selected structures (default: 'None')", default=None)
    parser.add_argument("-o"  , "--output"          , type=str     , required=True , **argv, help="output file with the selected structures")
    parser.add_argument("-of" , "--output_format"   , type=str     , required=False, **argv, help="output file format (default: 'None')", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    frames = read(args.input, index=':', format=args.input_format)  #eV
    print("done")

    # print("\n\tConverting the atomic structures positions and cells from 'atomic unit' to 'angstrom' ... ",end="")
    # factor = convert(1,"length","atomic_unit","angstrom")
    # for n in range(len(frames)):
    #     frames[n].positions *= factor
    #     if np.any(frames[n].pbc):
    #         cell = factor * frames[n].get_cell()
    #         frames[n].set_cell(cell)
    # print("done")

    # available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
    # available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

    # print('\tNumber of frames: ', len(frames))
    # print('\tNumber of atoms/frame: ', len(frames[0]))
    # print('\tAvailable structure properties: ', available_structure_properties)
    # print('\tAvailable atom-level properties: ', available_atom_level_properties)

    # if args.soap_hyper is not None :
    #     print("\n\tReading the SOAP hyperparameters from file '{:s}' ... ".format(args.soap_hyper),end="")
    #     with open(args.soap_hyper, 'r') as file:
    #         # Load the JSON data from the file
    #         user_soap_hyper = json.load(file)
    #     print("done")
    # else:
    #     user_soap_hyper = None

    # print("\n\tUsing the following SOAP hyperparameters:")
    # SOAP_HYPERS = {
    #     "interaction_cutoff": 3.5,
    #     "max_radial": 8,
    #     "max_angular": 6,
    #     "gaussian_sigma_constant": 0.4,
    #     "cutoff_smooth_width": 0.5,
    #     "gaussian_sigma_type": "Constant",
    # }

    # SOAP_HYPERS = add_default(user_soap_hyper,SOAP_HYPERS)
    # # SOAP_HYPERS["interaction_cutoff"] = args.cutoff_radius
    # show_dict(SOAP_HYPERS,string="\t\t")

    # #
    # print("\n\tPreparing SOAP object ... ",end="")
    # numbers = list(sorted(set([int(n) for frame in frames for n in frame.numbers])))

    # # initialize SOAP
    # soap = SOAP(
    #     global_species=numbers,
    #     expansion_by_species_method='user defined',
    #     **SOAP_HYPERS
    # )
    # print("done")

    # X = None
    # print("\tComputing SOAP features ... ")
    # for i, frame in enumerate(tqdm(frames)):
    #     # normalize cell for librascal input
    #     if np.linalg.norm(frame.cell) < 1e-16:
    #         extend = 1.5 * (np.max(frame.positions.flatten()) - np.min(frame.positions.flatten()))
    #         frame.cell = [extend, extend, extend]
    #         frame.pbc = True
    #     frame.wrap(eps=1e-16)

    #     x = soap.transform(frame).get_features(soap).mean(axis=0) # here it takes mean over atoms in the frame
    #     if X is None:
    #         X = np.zeros((len(frames), x.shape[-1]))
    #     X[i] = x

    # print(f"\n\tSOAP features shape: {X.shape}")

    print("\tReading SOAP descriptors from file '{:s}' ... ".format(args.soap_descriptors),end="")
    if str(args.soap_descriptors).endswith("npy"):
        X = np.load(args.soap_descriptors)
    elif str(args.soap_descriptors).endswith("txt"):
        X = np.loadtxt(args.soap_descriptors)
    print("done")

    #
    print("\tExtracting structures using the FPS algorithm:")
    struct_idx = FPS(n_to_select=args.number, progress_bar = True, initialize = 'random').fit(X.T).selected_idx_
    X_fps = X[struct_idx]

    print("\n\tFPS selected indices: {:d}".format(struct_idx.shape[0]))
    print(f"\tOriginal: {X.shape} ---> FPS: {X_fps.shape}")

    indices = np.asarray([ int(i) for i in struct_idx],dtype=int)

    if args.sort:
        print("\n\tSorting indices ... ",end="")
        indices = np.sort(indices)
        print("done")

    # Saving the fps selected structure
    if args.output_indices :
        print("\n\tSaving indices of selected structures to file '{:s}' ... ".format(args.output_indices),end="")
        np.savetxt(args.output_indices,indices,fmt='%d')
        print("done")

    # Saving the fps selected structure
    if args.output is not None :
        print("\n\tSaving FPS selected structures to file '{:s}' ... ".format(args.output),end="")
        frames_fps = [frames[i] for i in indices]
        write(args.output, frames_fps, format=args.output_format) # fmt)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
