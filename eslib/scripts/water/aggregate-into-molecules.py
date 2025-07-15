#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog
from eslib.input import slist
from eslib.mathematics import melt

#---------------------------------------#
# Description of the script's purpose
description = "Aggregate structures into molecules."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str     , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str     , help="molecule name (default: %(default)s)", default="molecule")
    parser.add_argument("-p" , "--properties"   , **argv, required=False, type=slist   , help="properties to aggregate (default: %(default)s)", default=None)
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str     , help="CSV output file (default: %(default)s)", default="agg-molecules.csv")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading atomic structures from file '{args.input}'"):
        trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("\t Number atomic structures: ",len(trajectory))
    
    # print("\n\tTrajectory summary: ")
    # df = trajectory.summary()
    # tmp = "\n"+df.to_string(index=False)
    # print(tmp.replace("\n", "\n\t"))
    
    #------------------#
    with eslog(f"\nExtracting molecules index from '{args.molecule}'"):
        molecules = trajectory.get(args.molecule)
        
    #------------------#
    assert np.all(molecules == molecules[0]), "error"
    molecule_0 = np.asarray(molecules[0])
    assert molecule_0.ndim == 1, "error"
    assert len(molecule_0) == trajectory.num_atoms(), "error"
    
    #------------------#
    extra_properties = {}
    if args.properties is not None:
        for prop in args.properties:
            with eslog(f"Extracting '{prop}' from the trajectory"):
                extra_properties[prop] = trajectory.get(prop,what="arrays")
    
    #------------------#
    u,c = np.unique(molecule_0,return_counts=True)
    N_atom_per_molecule = c[0]
    assert np.all(c==N_atom_per_molecule), "All the molecules should have the same number of atoms"
    
    #------------------#
    with eslog(f"\nExtracing masses"):
        pos = trajectory.get("positions")
        masses = np.asarray(trajectory.call(lambda a: a.get_masses()))
        pos = pos * masses[:,:,None]
        molecule_mass = masses[0,molecule_0 == u[0]].sum()
        pos /= molecule_mass
    
    print(f"\t Mass of one molecules: {molecule_mass:.2f} amu")
    
    #------------------#
    with eslog(f"\nConstructing dataset with atoms positions"):
        df = melt(pos,{0:"time",1:"atom"},["Rx","Ry","Rz"])
        
    with eslog(f"Constructing dataset with the molecules indices"):
        mol = melt(molecules[:,:,None],{0:"time",1:"atom"},["molecule"])
        
    with eslog(f"Merging datasets with positions and the molecules indices"):
        df = pd.merge(df,mol,on=["time","atom"])        
        
    #------------------#
    for key,prop in extra_properties.items():
        with eslog(f"Constructing dataset for '{key}'"):
            prop = np.asarray(prop)
            if prop.ndim == 2:
                prop = prop[:,:,None]
            assert prop.ndim == 3, "error"
            names = [f"{key}_{i}" for i in range(prop.shape[2])]
            prop_df = melt(prop,{0:"time",1:"atom"},names)   
            
        with eslog(f"Merging this dataset to the general one"):  
            df = pd.merge(df,prop_df,on=["time","atom"])    
        
    #------------------#
    with eslog(f"Aggregating over the molecules"):
        columns = [col for col in df.columns if col not in ['time', 'molecule','atom']]
        df = df.groupby(['time', 'molecule'])[columns].sum().reset_index()
        
    print("\n\t Final dataframe information:")
    print("\t - shape: ",df.shape)
    print("\t - columns: ",list(df.columns))
    print(f"\t - memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.3f} MB")
    # string = df.head().__repr__()
    # string = string.replace("\n","\n\t\t")
    # print("\t - head:")
    # print(string)
    
    #------------------#
    with eslog(f"\nSaving results dataframe to '{args.output}'"):
        df.to_csv(args.output,index=False)
        
    print("\n\t If you want to prettify the output file consider using:")
    print(f"\n\t column -s, -t {args.output} > prettified.txt")
    
    return

#---------------------------------------#
if __name__ == "__main__":
    main()


