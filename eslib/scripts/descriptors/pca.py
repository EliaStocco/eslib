#!/usr/bin/env python
from ase.io import read
import chemiscope
import numpy as np
import json
from skmatter.preprocessing import StandardFlexibleScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from eslib.formatting import esfmt
from eslib.functions import add_default
from eslib.show import show_dict

#---------------------------------------#
# Description of the script's purpose
description = "Process atomic structures and select a diverse subset using the Farthest Point Sampling (FPS) algorithm."

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-x"  , "--soap_descriptors", type=str     , required=True , **argv, help="file with the SOAP descriptors")


    parser.add_argument("-i" , "--input"             , type=str  , **argv, help="input file with the atomic structures [au]")
    # parser.add_argument("-o"  , "--output"            , type=str  , **argv, help="output file with the selected structures")
    # parser.add_argument("-oi" , "--output_indices"    , type=str  , **argv, help="output file with indices of the selected structures (default: %(default)s)",default='indices.txt')
    parser.add_argument("-if" , "--input_format"      , type=str  , **argv, help="input file format (default: %(default)s)" , default=None)
    # parser.add_argument("-of" , "--output_format"     , type=str  , **argv, help="output file format (default: %(default)s)", default=None)
    # parser.add_argument("-ff" , "--features_file"     , type=str  , **argv, help="features output file [*.npy] (default: %(default)s)", default=None)
    # parser.add_argument("-n"  , "--number"            , type=int  , **argv, help="number of desired structure (default: %(default)s)", default=100)
    # parser.add_argument("-s"  , "--sort"              , type=str2bool , **argv, help="whether to sort the indices (default: %(default)s)", default=True)
    # parser.add_argument("-rc" , "--cutoff_radius"     , type=float, **argv, help="cutoff radius [au] (default: %(default)s)", default=6)
    # parser.add_argument("-sh" , "--soap_hyper"        , type=str  , **argv, help="JSON file with the SOAP hyperparameters", default=None)
    # parser.add_argument("-pca", "--pca"               , type=str2bool  , **argv, help="whether to perform PCA or not (default: %(default)s)", default=True)
    parser.add_argument("-p" , "--parameters" , type=str  , **argv, help="JSON file with the parameters for the sklearn method (default: %(default)s)", default=None)
    parser.add_argument("-j", "--component" , type=int  , **argv, help="cartesian component of the features to be saved (default: %(default)s)", default=0)
    parser.add_argument("-m", "--method"    , type=str  , **argv, help="analysis method (default: %(default)s)", default=2, choices=['pca','tsne','mds'])
    parser.add_argument("-n", "--number"    , type=int  , **argv, help="number of components to be used in PCA (default: %(default)s)", default=2)
    parser.add_argument("-f", "--feature"   , type=str  , **argv, help="feature to be analysed using PCA (default: %(default)s)", default="dipole")
    parser.add_argument("-o", "--output"    , type=str  , **argv, help="output file for PCA (default: %(default)s)", default="pca.txt")
    parser.add_argument("-c", "--chemiscope", type=str  , **argv, help="output file for chemiscope (default: %(default)s)", default="chemiscope.json.gz")
    return parser# .parse_args()

#---------------------------------------#
def save_feature_to_chemiscope(features,name,frames,T,file):
    properties = {
        "PCA": {
            # change the following line if your map is per-atom
            "target": "structure",
            "values": T,

            # change the following line to describe your map
            "description": "PCA of structure-averaged representation",
        },

        # this is an example of how to add structure-level properties
        name: {
            "target": "structure",
            "values": np.asarray(features).flatten().tolist(),

            # change the following line to correspond to the units of your property
            "units": "atomic_unit",
        },

        # # this is an example of how to add atom-level properties
        "numbers": {
            "target": "atom",
            "values": np.concatenate([frame.arrays['numbers'] for frame in frames]),
        },
    }

    chemiscope.write_input(
        path=file,
        frames=frames,
        properties=properties,

        # # This is required to display properties with `target: "atom"`
        # # Without this, the chemiscope will show only structure-level properties
        # environments=chemiscope.all_atomic_environments(frames),
    )

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    frames = read(args.input, index=':', format=args.input_format)  #eV
    print("done")

    available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
    available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

    print("\tAvailable structure-level properties:", available_structure_properties)
    print("\t     Available atom-level properties:", available_atom_level_properties)
    
    #------------------#
    print("\n\tReading SOAP descriptors from file '{:s}' ... ".format(args.soap_descriptors),end="")
    if str(args.soap_descriptors).endswith("npy"):
        X = np.load(args.soap_descriptors)
    elif str(args.soap_descriptors).endswith("txt"):
        X = np.loadtxt(args.soap_descriptors)
    print("done")

    #------------------#
    # Visualizing using the principal componenets of the selected dataset and the original dataset using the chemiscope
    print("\n\tStandardyzing features ... ",end="")
    X = StandardFlexibleScaler(column_wise=False).fit_transform(X)
    print("done")

    #------------------#
    user_par = None
    if args.parameters is not None :
        print("\n\tReading parameters from file '{:s}' ... ".format(args.parameters),end="")
        with open(args.parameters, 'r') as file:
            # Load the JSON data from the file
            user_par = json.load(file)
        print("done")
        print("\tReading parameters:",end="")
        show_dict(user_par,string="\t")

    print("\tApplying {:s} ... ".format(args.method),end="")
    if args.method == "pca":
        default_par = {}
        parameters = add_default(user_par,default_par)
        T = PCA(n_components=args.number,**parameters).fit_transform(X)
    elif args.method == "tsne":
        default_par = {'learning_rate':'auto', 'init':'random', 'perplexity':40}
        parameters = add_default(user_par,default_par)
        T = TSNE(n_components=args.number,**parameters).fit_transform(X)
    elif args.method == "mds":
        default_par = {}
        parameters = add_default(user_par,default_par)
        T = MDS(n_components=args.number,**parameters).fit_transform(X)
    print("done")

    #------------------#
    print("\tExtracting '{:s}' from the atomic structures ... ".format(args.feature),end="")
    try :
        features = np.array([frame.info[args.feature] for frame in frames])
    except:
        raise ValueError("encountered problems extracting '{:s}' info from the atomic structures".format(args.feature))
    print("done")
    print("\tfeatures.shape: ",features.shape)

    tmp = np.concatenate([T, features.reshape(len(features),-1)], axis=1)
    print("\tSaving analysis results to file '{:s}' ... ".format(args.output),end="")
    np.savetxt(args.output,tmp,fmt='%24.18e')
    print("done")

    print()
    extension = ".json.gz"
    base_name = str(args.chemiscope).split(extension)[0]
    if features.ndim == 1 and args.component >= 0 :
        raise ValueError("`features` is a scalar, only `-j 0` allowed.")
    
    if features.ndim > 1:
        if args.component == -1:
            for n in range(features.shape[1]):
                file = f"{base_name}.{n}{extension}"
                print("\tSaving results (component {:d}) for chemiscope to file '{:s}' ... ".format(n,file),end="")
                save_feature_to_chemiscope(features[:,n],args.feature,frames,T,file)
                print("done")
        elif args.component == -2:
            file = f"{base_name}{extension}"
            print("\tSaving results (modulus) for chemiscope to file '{:s}' ... ".format(file),end="")
            features = np.linalg.norm(features,axis=1)
            save_feature_to_chemiscope(features,args.feature,frames,T,file)
            print("done")
        else:
            file = f"{base_name}{extension}"
            print("\tSaving results for chemiscope to file '{:s}' ... ".format(file),end="")
            save_feature_to_chemiscope(features[:,args.component],args.feature,frames,T,file)
            print("done")
    else:
        file = f"{base_name}{extension}"
        print("\tSaving results for chemiscope to file '{:s}' ... ".format(file),end="")
        save_feature_to_chemiscope(features,args.feature,frames,T,file)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
