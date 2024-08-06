#!/usr/bin/env python
import numpy as np
import os
from eslib.nn.user import get_model
from eslib.functions import suppress_output
from eslib.input import str2bool
from classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, warning
import torch 
from tqdm import tqdm
from eslib.nn.dataset import make_dataloader
from eslib.scripts.nn.dataset2extxyz import Data2Atoms

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dipole values for a trajectory using a neural network."

#---------------------------------------#
def prepare_args(description):
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--instructions", type=str     , **argv, help="model input file (default: %(default)s)", default="instructions.json")
    parser.add_argument("-p" , "--parameters"  , type=str     , **argv, help="torch parameters file (default: %(default)s)", default="parameters.pth",)
    parser.add_argument("-t" , "--trajectory"  , type=str     , **argv, help="trajectory file [a.u.]")
    parser.add_argument("-z" , "--compute_BEC" , type=str2bool, **argv, help="whether to compute BECs (default: %(default)s)", default=False)
    # parser.add_argument("-d" , "--debug"       , type=str2bool, **argv, help="debug mode (default: %(default)s)", default=False)
    parser.add_argument("-o" , "--output"      , type=str     , **argv, help="output file with the dipoles (default: %(default)s)", default="dipole.nn.txt")
    parser.add_argument("-oz", "--output_BEC"  , type=str     , **argv, help="output file with the BECs (default: %(default)s)", default="bec.nn.txt")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\t{:s}: this script has to be optimized for large trajectories.".format(warning))

    #------------------#
    # trajectory
    use_dataset = False
    if str(args.trajectory).endswith(".pth"):
        trajectory = torch.load(args.trajectory)   
        use_dataset = True
        # dataloader_train = make_dataloader(train_dataset, batch_size)  
    else:
        print("\tReading atomic structures from file '{:s}' ... ".format(args.trajectory), end="")
        trajectory = AtomicStructures.from_file(file=args.trajectory)
        print("done")

    N = len(trajectory)
    print("\tn. of atomic structures: {:d}".format(N))
    example = Data2Atoms(trajectory[0])[0] if use_dataset else trajectory[0]

    #------------------#
    print("\tLoading model ... ",end="")
    file_in = os.path.normpath("{:s}".format(args.instructions))
    file_pa = os.path.normpath("{:s}".format(args.parameters))
    with suppress_output():
        model = get_model(file_in,file_pa)
        model.store_chemical_species(example)
    print("done")

    #------------------#
    line = " and BEC" if args.compute_BEC else ""
    print("\tComputing dipole{:s} ... ".format(line))
    D = np.full((N,3),np.nan)
    if args.compute_BEC:
        Z = np.full((N,3*len(example.positions),3),np.nan)

    if not use_dataset:
        #with tqdm(enumerate(trajectory)) as bar:
        for n,atoms in enumerate(list(trajectory)):
            print("\t{:>6d}/{:<d}".format(n+1,N),end="\r")
            # pos = atoms.positions
            # cell = np.asarray(atoms.cell).T if np.all(atoms.get_pbc()) else None
            if args.compute_BEC:
                d,z,x = model.get_value_and_jac(pos=atoms.get_positions(),cell=atoms.get_cell())
                Z[n,:,:] = z.detach().numpy()#.flatten()
            else:
                d,x = model.compute(pos=atoms.get_positions(),cell=atoms.get_cell())
            D[n,:] = d.detach().numpy()#.flatten()
    else:        
        if args.compute_BEC:
            for n,X in tqdm(enumerate(trajectory),total=len(trajectory)):
                d,z,x = model.get_value_and_jac(X=X)
                Z[n,:,:] = z.detach().numpy()
                D[n,:] = d.detach().numpy()
        else:
            X = next(iter(make_dataloader(dataset=trajectory,batch_size=len(trajectory),shuffle=False,drop_last=False)))
            d,x = model.compute(X=X)
            D = d.detach().numpy()

    #------------------#
    print("\tSaving dipoles to file '{:s}' ... ".format(args.output),end="")
    np.savetxt(args.output,D)
    print("done")

    #------------------#
    if args.compute_BEC:
        print("\tSaving BECs to file '{:s}' ... ".format(args.output_BEC),end="")
        with open(args.output_BEC,"w") as f:
            for n in range(N):
                np.savetxt(f,Z[n,:,:],header="step: {:d}".format(n))
        print("done")

if __name__ == "__main__":
    main()
