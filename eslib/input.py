import numpy as np
import argparse
#---------------------------------------#
def union_type(s:str,dtype):
    return s
#---------------------------------------#
def size_type(s:str,dtype=int,N=None):
    s = s.replace("[","").replace("]","").replace(","," ").split()
    # s = s.split("[")[1].split("]")[0].split(",")
    if N is not None and len(s) != N :
        raise ValueError("You should provide {:d} values".format(N)) 
    else:
        return np.asarray([ dtype(k) for k in s ])
    
flist = lambda s:size_type(s,float) # float list
ilist = lambda s:size_type(s,int)   # integer list
slist = lambda s:size_type(s,str)   # string list
#---------------------------------------#
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")