import argparse
import ast
import re

import numpy as np
from typing import Union


#---------------------------------------#
def itype(x:str):
    return int(x) if x.isdigit() else x

#---------------------------------------#
def literal(value):
    return ast.literal_eval(value)

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
        # if dtype != str:
        #     return np.asarray([ dtype(k) for k in s ])
        # else:
        #     return list([ dtype(k) for k in s ])
    
flist = lambda s:size_type(s,float) # float list
ilist = lambda s:size_type(s,int)   # integer list
slist = lambda s:size_type(s,str)   # string list

#---------------------------------------#
def str2bool(v:Union[bool,str]):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
blist = lambda s: size_type(s, str2bool)
    
    
#---------------------------------------#
# Function to parse nested lists of specific data types
def nested_list_type(s: str, dtype=int):
    # Remove outer brackets, split by '],[' and then process each inner list
    s = s.strip().replace(" ", "")  # Remove all whitespace
    # Regex to extract the list components
    s = re.findall(r'\[(.*?)\]', s)
    
    # Parse the string into nested lists
    parsed = [size_type(inner, dtype=dtype) for inner in s]
    
    return np.array(parsed)

# nested int list
nilist = lambda s: nested_list_type(s, int)
nslist = lambda s: nested_list_type(s, str)
