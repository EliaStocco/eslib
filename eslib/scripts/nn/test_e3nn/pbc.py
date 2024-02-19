#!/usr/bin/env python
import numpy as np
import os
from eslib.nn.user import get_model
from ase.io import read
from eslib.formatting import esfmt
from eslib.tools import frac2cart
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Check the E(3)-equivariance of a neural network."

def prepare_args(description):
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--instructions", type=str, **argv, help="model input file (default: 'instructions.json')", default="instructions.json")
    parser.add_argument("-p" , "--parameters"  , type=str, **argv, help="torch parameters file (default: 'parameters.pth')", default=None)
    parser.add_argument("-s" , "--structure"   , type=str, **argv, help="file with an atomic structure [a.u.]")
    parser.add_argument("-n" , "--number"      , type=int, **argv, help="number of tests to perform", default=100)
    parser.add_argument("-f" , "--fold"        , type=str2bool, **argv, help="whether the atomic structures have to be folded into the primitive unit cell (default: false)", default=False)
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.structure), end="")
    atoms = read(args.structure)
    print("done")

    #------------------#
    pbc = np.all(atoms.get_pbc())
    if not pbc:
        return 0.

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
    pos = atoms.positions.reshape((-1,3))
    cell = atoms.get_cell()

    #------------------#
    print("\n\tGenerating {:d} random translation vectors ... ".format(args.number),end="")
    zeros = np.zeros(args.number)
    MaxDisplacement = 4
    allTdir = {
        "1"   : np.concatenate([np.random.uniform(0, MaxDisplacement, args.number),zeros,zeros]).reshape((args.number,3)),
        "2"   : np.concatenate([zeros,np.random.uniform(0, MaxDisplacement, args.number),zeros]).reshape((args.number,3)),
        "3"   : np.concatenate([zeros,zeros,np.random.uniform(0, MaxDisplacement, args.number)]).reshape((args.number,3)),
        "all" : np.concatenate([np.random.uniform(0, MaxDisplacement, args.number),\
                                np.random.uniform(0, MaxDisplacement, args.number),\
                                np.random.uniform(0, MaxDisplacement, args.number)]).reshape((args.number,3))
    }
    print("done")

    #------------------#
    print("\tComparing 'outputs from rotated inputs' with 'rotated outputs' ... ",end="")
    y,_ = model.get(pos=pos.reshape((-1,3)),cell=cell)
    y = y.detach().numpy()
    shape = (len(allTdir.keys()),*allTdir["1"].shape)
    norm = np.zeros(shape)
    k = 0
    for key,allT in allTdir.items():
        for n,T in enumerate(allT):
            # Translated input (x) to output (y)
            T = frac2cart(cell=cell,v=T)
            newpos = pos + T
            Tx2y, _ = model.get(pos=newpos,cell=cell) 
            Tx2y = Tx2y.detach().numpy()
            norm[k,n] = np.linalg.norm(Tx2y - y)
        k += 1
    print("done")
    
    print("\tSummary of the norm between 'outputs from translated inputs' and 'outputs'")
    print("\t{:>20s}: {:.4e}".format("min norm",norm.min()))
    print("\t{:>20s}: {:.4e}".format("max norm",norm.max()))
    print("\t{:>20s}: {:.4e}".format("mean norm",norm.mean()))

    return norm

#---------------------------------------#
if __name__ == "__main__":
    main()
