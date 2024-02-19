import numpy as np
import os
from ase.io import read
from eslib.nn.user import get_model

def get_tools(args):
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
    pbc = np.all(atoms.get_pbc())
    pos = atoms.positions
    cell = np.asarray(atoms.cell) if np.all(atoms.get_pbc()) else np.full((3,3),np.nan)   
    def func(array):
        if pbc:
            cell = array[0:3,:].T
            pos  = array[3:,:]
        else:
            cell = None
            pos = array
        y,_ = model.get(pos=pos.reshape((-1,3)),cell=cell)
        return y.detach().numpy()
    
    if pbc:
        array = np.concatenate([cell,pos])
    else:
        array = pos

    return {"model":model,"func":func,"array":array,"atoms":atoms,"pbc":pbc} 