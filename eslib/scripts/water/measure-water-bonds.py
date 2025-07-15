#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog
from eslib.mathematics import melt

#---------------------------------------#
# Description of the script's purpose
description = "Measure the water bonds length."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str     , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str     , help="molecule name (default: %(default)s)", default="molecule")
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str     , help="output file (default: %(default)s)", default="O-H.dist.csv")
    parser.add_argument("-p" , "--plot"         , **argv, required=False, type=str     , help="output plot file (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    print("\tNumber atomic structures: ",len(trajectory))
    
    #------------------#
    with eslog(f"\nExtracting molecules index from '{args.molecule}'"):
        molecules = trajectory.get(args.molecule)
        
    #------------------#
    assert np.all(molecules == molecules[0]), "error"
    molecule_0 = np.asarray(molecules[0])
    assert molecule_0.ndim == 1, "error"
    assert len(molecule_0) == trajectory.num_atoms(), "error"
    
    pos = trajectory.get("positions")
    
    chemical = trajectory.call(lambda x:x.get_chemical_symbols())
    chemical = np.asarray(chemical)
    
    #------------------#
    with eslog(f"\nConstructing dataset with atoms positions"):
        df = melt(pos,{0:"structure",1:"atom"},["Rx","Ry","Rz"])
        
    with eslog(f"Constructing dataset with the molecules indices"):
        mol = melt(molecules[:,:,None],{0:"structure",1:"atom"},["molecule"])
        
    with eslog(f"Constructing dataset with the chemical species"):
        chem = melt(chemical[:,:,None],{0:"structure",1:"atom"},["chemical"])
        
    with eslog(f"Merging datasets with positions and the molecules indices"):
        df = pd.merge(df,mol,on=["structure","atom"])  
        df = pd.merge(df,chem,on=["structure","atom"]) 
        
    #------------------#
    with eslog(f"Computing O-H distances"):
    
        # Split into H and O DataFrames
        df_H = df[df['chemical'] == 'H'].copy()
        df_O = df[df['chemical'] == 'O'].copy()

        # Merge H with corresponding O based on structure and molecule
        df_merged = pd.merge(
            df_H, df_O,
            on=['structure', 'molecule'],
            suffixes=('_H', '_O')
        )

        # Compute 3D distance between each H and its corresponding O
        df_merged['distance'] = np.sqrt(
            (df_merged['Rx_H'] - df_merged['Rx_O'])**2 +
            (df_merged['Ry_H'] - df_merged['Ry_O'])**2 +
            (df_merged['Rz_H'] - df_merged['Rz_O'])**2
        )
        
        df_result = df_merged[['structure', 'molecule', 'distance']].copy()
        
    #------------------#
    distances = np.asarray(df_result['distance'])
    dmean, dmin, dmax = np.mean(distances), np.min(distances), np.max(distances)
    print("\n\t Statistics:")
    print("\t - mean:",dmean)
    print("\t -  min:",dmin)
    print("\t -  max:",dmax)
    
    #------------------#
    with eslog(f"\nSaving results dataframe to '{args.output}'"):
        df_result.to_csv(args.output,index=False)

    if args.plot is not None:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.hist(df_result['distance'], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('O-H bond length')
        plt.ylabel('Frequency')
        plt.title('Histogram of O-H bond lengths')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"\n\t Histogram saved to '{args.plot}'")    
    
#---------------------------------------#
if __name__ == "__main__":
    main()


