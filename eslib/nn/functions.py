import subprocess
from typing import Tuple

import numpy as np
import torch

from eslib.functions import add_default


def vectorize(A:callable,min_shape=1,init=torch.zeros):
    def B(x):
        x = np.asarray(x)
        if len(x.shape) > min_shape :
            N = len(x)
            tmp = A(x[0])
            out = init((N,*tmp.shape))
            out[0,:] = tmp
            for n in range(1,N):
                out[n,:] = B(x[n])
            return out
        else : 
            return A(x)
    return B


def get_type_onehot_encoding(species:list)->Tuple[torch.Tensor,dict]:
    type_encoding = {}
    for n,s in enumerate(species):
        type_encoding[s] = n
    type_onehot = torch.eye(len(type_encoding))
    return type_onehot, type_encoding 
 

@vectorize
def symbols2x(symbols): 
    symbols = np.asarray(symbols)
    # `np.unique` returns the sorted unique values of an array!
    # We need species to be sorted! 
    # Any kind of sorting is okay, as long as the ouput does not depend on order on the input elements.
    # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    species = np.unique(symbols)
    type_onehot, type_encoding = get_type_onehot_encoding(species)
    return type_onehot[[type_encoding[atom] for atom in symbols]]

def get_data_from_dataset(dataset,variable):
    # Extract data for the specified variable from the dataset
    v = getattr(dataset[0],variable)
    data = torch.full((len(dataset),*v.shape),np.nan)
    for n,x in enumerate(dataset):
        data[n,:] = getattr(x,variable)
    return data

def bash_as_function(script_path,opts=None):
    """run a bash script by calling this function"""
    default = {
        "print" : False
    }
    opts = add_default(opts,default)
    def wrapper():
        try:
            # Run the Bash script using subprocess
            completed_process = subprocess.run(
                ['bash', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,  # Set this to True if you want to use shell features like piping
                env=None,  # Use the current environment
            )

            # Check the return code to see if the script executed successfully
            if completed_process.returncode == 0:
                if opts["print"] : print("Script executed successfully.")
                if opts["print"] : print("Script output:")
                if opts["print"] : print(completed_process.stdout)
            else:
                if opts["print"] : print("Script encountered an error.")
                if opts["print"] : print("Error output:")
                if opts["print"] : print(completed_process.stderr)
        except:
            print("Error running script '{:s}'".format(script_path))            
        return
    return wrapper