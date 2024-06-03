#!/usr/bin/env python
from ase.io import read
from eslib.formatting import esfmt, warning
from ase import Atoms
import numpy as np
from ase.io import read, write
from eslib.tools import cart2frac, frac2cart
from eslib.input import slist
from typing import List

#---------------------------------------#
# Description of the script's purpose
description = "Wrap hydrogen atoms such that they will be close to an oxygen atoms."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str  , help="file with the atomic structures")
    parser.add_argument("-if", "--input_format" , **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    # parser.add_argument("-rc", "--cutoff"       , **argv, required=False, type=float, help="cutoff/bond length(default: %(default)s)" , default=3)
    parser.add_argument("-n" , "--n_bonds"      , **argv, required=False, type=int  , help="number of bonds (default: %(default)s)", default=2)
    parser.add_argument("-s" , "--species"      , **argv, required=False, type=slist, help="atomic species of the bonds to be fixed (default: %(default)s)", default=['O','H'])
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str  , help="output file with the oxidation numbers (default: %(default)s)", default="wrapped.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str  , help="output file format (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    if len(args.species) != 2:
        raise ValueError("-s,--species has to be of length 2.")
    print("\tFixing bonds between {:s} and {:s}".format(args.species[0],args.species[1]))
    
    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory:List[Atoms] = read(args.input,format=args.input_format,index=":")
    print("done")
    print("\tNumber atomic structures: ",len(trajectory))

    #------------------#
    # Select oxygen and hydrogen atoms
    oxygens   = [ n for n,a in enumerate(trajectory[0]) if a.symbol == args.species[0]]
    hydrogens = [ n for n,a in enumerate(trajectory[0]) if a.symbol == args.species[1]]
    print()
    print("\tNumber of {:s} atoms: {:d}".format(args.species[0],len(oxygens)))
    print("\tNumber of {:s} atoms: {:d}".format(args.species[1],len(hydrogens)))


    #------------------#
    print("\tFixing bonds:")
    zeros = np.zeros(3)
    N = len(trajectory)
    for n,atoms in enumerate(trajectory):
        print("\t - atomic structure {:d}/{:d}".format(n+1,N),end="\r")
        oxygens   = [ n for n,a in enumerate(trajectory[0]) if a.symbol == args.species[0]]
        hydrogens = [ n for n,a in enumerate(trajectory[0]) if a.symbol == args.species[1]]

        wrapped = []
        for o_index in oxygens:
            # Find neighbors of the current oxygen atom within the cutoff distance
            
            distances = atoms.get_distances(o_index,hydrogens,mic=True,vector=False)
            indices = list(np.argsort(distances)[:args.n_bonds])

            for n in np.asarray(hydrogens)[indices]:
                # if d > args.cutoff:
                #     continue
                delta = atoms.positions[n] - atoms.positions[o_index] 
                delta:np.ndarray = cart2frac(atoms.get_cell(),delta)
                # delta = (2*delta).round(0)/2.
                delta = delta.round(0).astype(int)
                if not np.allclose(delta,zeros):
                    # print("\t - wrapping hydrogen {:3d} by [{:>2d},{:>2d},{:>2d}]".format(n,*delta.tolist()),\
                    #     "(frac. coor.) so that it will be closer to oxygen {:d}".format(o_index))
                    delta = frac2cart(atoms.get_cell(),delta)
                    atoms.positions[n,:] -= delta
                    wrapped.append(n)
                    # print(n," ",delta)
            

        Nwrapping = len(wrapped)
        Nwrapped  = len(np.unique(wrapped))
        # print("\n\tNumber of wrapping: ",Nwrapping)
        # print("\tNumber of wrapped hydrogens: ",Nwrapped)

        if Nwrapping != Nwrapped:
            print("\t{:s}: the previous two numbers are expected to be the same. Carefully check your input and output files.".format(warning))


    print("\n\tWriting (un)wrapped atomic structure to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=trajectory,filename=args.output,format=args.output_format) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

#---------------------------------------#
if __name__ == "__main__":
    main()

# {
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/eslib/scripts/convert/fix-water-bonds.py",
#             "cwd" : "/home/stoccoel/google-personal/works/water/MACE/",
#             "console": "integratedTerminal",
#             "justMyCode": false,
#             "args": [
#                 "-i", "MACE_test.extxyz", 
#                 // "-m", "model/dipole_32.model", 
#                 // "-p", "your_port_number", 
#                 // "-a", "your_address", 
#                 // "-d", "cpu"
#             ],
#             "env": {
#                 "PYTHONPATH": "/home/stoccoel/google-personal/codes/mace/mace/"
#             }
#         }
#     ]
# }


