import torch
import numpy as np
from typing import Tuple

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
 
# #@np.vectorize
# def symbols2x(symbols): 
#     symbols = np.asarray(symbols)
#     if len(symbols.shape) > 1 :
#         return np.vectorize(symbols2x)(symbols)
#     else :
#         species = np.unique(symbols)
#         type_onehot, type_encoding = get_type_onehot_encoding(species)
#         return type_onehot[[type_encoding[atom] for atom in symbols]]

#@np.vectorize(signature="(m,n)->(m)")
@vectorize
def symbols2x(symbols): 
    symbols = np.asarray(symbols)
    species = np.unique(symbols)
    type_onehot, type_encoding = get_type_onehot_encoding(species)
    return type_onehot[[type_encoding[atom] for atom in symbols]]

#symbols2x = np.vectorize(symbols2x, signature="(m)->()",otypes=[float])