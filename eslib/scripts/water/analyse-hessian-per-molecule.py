#!/usr/bin/env python
import numpy as np
import pandas as pd
from collections import defaultdict
from ase.io import read
from eslib.formatting import esfmt, float_format, eslog
from eslib.geometry import mic_dist

#---------------------------------------#
# Description of the script's purpose
description = "Analyse the results from 'aggregate-into-molecules.py'."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-m" , "--molecular_data"        , **argv, required=True , type=str  , help="csv input file")
    parser.add_argument("-d" , "--displacement"      , **argv, required=False, type=float  , help="displacement value (default: %(default)s)", default=0.01)
    parser.add_argument("-i" , "--input"         , **argv, required=True, type=str  , help="input file [extxyz]")
    parser.add_argument("-o" , "--output"       , **argv, required=False, type=str  , help="output file (default: %(default)s)", default="hessian-per-molecule.csv")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    with eslog(f"Reading data fram file '{args.molecular_data}'"):
        df = pd.read_csv(args.molecular_data)
        df = df.rename(columns={"time": "displaced-atom"})
        
    snapshots = df["displaced-atom"].unique()
    assert len(snapshots) == 19, "The input file should contain 19 snapshot"
    
    #------------------#
    print("\n\tPreparing dataframes ... ",end="")
    original = df[df["displaced-atom"] == 0]
    del original["displaced-atom"]
    
    plus  = df[df["displaced-atom"].isin(np.arange(0,9)*2+1).tolist()].reset_index()
    minus = df[df["displaced-atom"].isin(np.arange(0,9)*2+2).tolist()].reset_index()
    assert len(plus) == 9*len(original), "error"
    assert len(minus) == 9*len(original), "error"
    
    cols = plus.columns.tolist()
    for k in ["index","Rx","Ry","Rz"]:
        cols.remove(k)
    
    plus = plus[cols]
    minus = minus[cols]
    
    minus["displaced-atom"] -= 1
    
    plus = plus.set_index(["displaced-atom", "molecule"])
    minus = minus.set_index(["displaced-atom", "molecule"])
    
    delta:pd.DataFrame = (plus - minus)/(2*args.displacement)
    delta = delta.reset_index()
    
    original_pos = original[original["molecule"] == 0 ][["Rx","Ry","Rz"]].to_numpy().flatten()
    assert original_pos.shape == (3,), "error"
    
    original = original.set_index("molecule")
    print("done")
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input, index=0)
    print("done")
    
    #------------------#
    print("\tComputing distances ... ",end="")
    # delta["displaced-atom"] = (delta["displaced-atom"]-1)/2
    # delta["displaced-atom"] = delta["displaced-atom"].astype(np.int32)
    molecule_pos = original.loc[delta["molecule"],["Rx","Ry","Rz"]].to_numpy()
    delta_pos = molecule_pos - original_pos
    distance = mic_dist(delta_pos,atoms.get_cell())[1]
    distance = pd.DataFrame(columns=["distance"],data=distance)
    distance["molecule"] = delta["molecule"].copy()
    distance["displaced-atom"] = delta["displaced-atom"].copy()
    print("done")
    
    #------------------#
    print("\tPreparing output dataframe ... ",end="")
    plus = plus.reset_index()
    df = pd.merge(plus,distance)
    df = df[df["distance"] > 0]
    df["displaced-atom"] = ((df["displaced-atom"]-1)/2).astype(np.int32)
    print("done")
    
    #------------------#
    print("\tLooking for stored quantities ... ",end="")
    cols = df.columns.tolist()
    for k in ["displaced-atom","molecule","distance"]:
        cols.remove(k)
    final = df[["displaced-atom","molecule","distance"]].copy()
    
    groups = defaultdict(list)

    for c in cols:
        prefix = c.rsplit("_", 1)[0]  # everything except last "_number"
        groups[prefix].append(c)

    groups = dict(groups)  # convert back to normal dict
    print("done")
    print("\tFound quantities: ",list(groups.keys()))
    
    #------------------#
    print("\tComputing norm ... ",end="")
    for key,cols in groups.items():
        tmp = df[cols].to_numpy()
        norm = np.linalg.norm(tmp,axis=1)
        final[key] = norm
    print("done")
    
    #------------------#
    print("\n\tFinal dataframe shape:",final.shape)
    print("\tFinal dataframe has the following columns:",final.columns.tolist())
    
    #------------------#
    print(f"\n\tSaving to file '{args.output}' ... ",end="")
    final.to_csv(args.output,index=False,float_format=float_format)
    print("done")
    
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()


# { 
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/codes/eslib/eslib/scripts/water/analyse-hessian-per-molecule.py",
#             "console": "integratedTerminal",
#             "justMyCode": false,
#             "stopOnEntry": false,
#             "cwd" : "/home/stoccoel/Documents/water-revPBE0+D3/long-range/results/",
#             "args" : ["-m", "molecules/molecules-n=0.csv","-i","structures/originals.extxyz"]
#         }
#     ]
# }