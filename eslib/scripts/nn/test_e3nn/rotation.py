#!/usr/bin/env python
import numpy as np
import os
from eslib.nn.user import get_model
from ase.io import read
from e3nn.o3 import rand_matrix
from eslib.formatting import esfmt
from eslib.input import str2bool
from ase.geometry import wrap_positions

#---------------------------------------#
# Description of the script's purpose
description = "Check the E(3)-equivariance of a neural network."

def prepare_args(description):
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--instructions", type=str, **argv, help="model input file (default: %(default)s)", default="instructions.json")
    parser.add_argument("-p" , "--parameters"  , type=str, **argv, help="torch parameters file (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--structure"   , type=str, **argv, help="file with an atomic structure [a.u.]")
    parser.add_argument("-n" , "--number"      , type=int, **argv, help="number of tests to perform", default=100)
    parser.add_argument("-f" , "--fold"        , type=str2bool, **argv, help="whether the atomic structures have to be folded into the primitive unit cell (default: %(default)s)", default=False)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.structure), end="")
    atoms = read(args.structure)
    print("done")

    #------------------#
    print("\tLoading model ... ",end="")
    file_in = os.path.normpath("{:s}".format(args.instructions))
    file_pa = os.path.normpath("{:s}".format(args.parameters)) if args.parameters is not None else None
    model = get_model(file_in,file_pa)
    print("done")

    #------------------#
    from eslib.nn.network import iPIinterface
    if not isinstance(model,iPIinterface):
        raise ValueError("'model' should be 'iPIinterface'.")

    #------------------#
    pbc = np.all(atoms.get_pbc())
    if not pbc:
        args.fold = False
    pos = atoms.positions
    cell = atoms.get_cell() if pbc else None
    def func(array):
        if pbc:
            cell = array[0:3,:].T
            pos  = array[3:,:].reshape((-1,3))
            if args.fold:
                pos = wrap_positions(positions=pos,cell=cell.T)
        else:
            cell = None
            pos = array.reshape((-1,3))
        y,_ = model.get(pos=pos,cell=cell,check=False)
        return y.detach().numpy()
    
    if pbc:
        array = np.concatenate([np.asarray(cell),pos])
    else:
        array = pos

    #------------------#
    print("\n\tGenerating {:d} random rotation matrices ... ".format(args.number),end="")
    allR = rand_matrix(args.number)
    print("done")

    print("\tComparing 'outputs from rotated inputs' with 'rotated outputs' ... ",end="")
    y = func(array)
    norm = np.zeros(len(allR))
    for n,R in enumerate(allR):
        R = R.numpy()
        tmp = ( R @ array.T ).T
        Rx2y = func(tmp) # Rotated input (x) to output (y)
        Ry = R @ y       # Rotated output (y)
        norm[n] = np.linalg.norm(Rx2y - Ry)
    print("done")
    
    print("\tSummary of the norm between 'outputs from rotated inputs' and 'rotated outputs'")
    print("\t{:>20s}: {:.4e}".format("min norm",norm.min()))
    print("\t{:>20s}: {:.4e}".format("max norm",norm.max()))
    print("\t{:>20s}: {:.4e}".format("mean norm",norm.mean()))

    return norm

#---------------------------------------#
if __name__ == "__main__":
    main()
