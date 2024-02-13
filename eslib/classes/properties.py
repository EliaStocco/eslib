import pandas as pd
import numpy as np
from copy import copy
from .functions import getproperty, get_property_header

class properties:

    def __init__(self,info:dict=None):

        self.header     = list()
        self.properties = dict()
        self.units      = dict()
        self.length     = 0 
        
        if info is not None:
            self.header     = info["header"]
            self.properties = info["properties"]
            self.units      = info["units"]

        length = None
        for k in self.header:
            if k not in self.properties:
                raise IndexError("'{:s}' is not a valid property.".format(k))
            self.properties[k] = np.asarray(self.properties[k])
            N = len(self.properties[k])
            if length is None:
                length = N
            elif length != N:
                raise ValueError("All the properties should have the same length.")
        self.length = length
        
        pass
    
    @classmethod
    def load(cls,file):
        header = get_property_header(file,search=True)
        properties,units = getproperty(file,header)
        # header  = get_property_header(file,search=False)
        info = {
            "header"     : header,
            "properties" : properties,
            "units"      : units
        }
        return cls(info=info)
    
    def __getitem__(self,index):
        if type(index) == str:
            return self.properties[index]
        elif type(index) == int:
            out = dict()
            for k in self.header:
                out[k] = self.properties[k][index]
            return out
        elif type(index) == slice:
            out = copy(self)
            for k in self.header:
                out.properties[k] = self.properties[k][index]
            return out
        else:
            raise TypeError("index type not allowed/implemented")
        
    def __len__(self):
        return self.length
    
    # def to_pandas(self):
    #     df = pd.DataFrame(columns=self.header,dtype=object)
    #     for k in self.heder:
    #         df[k] = self.properties[k]
    #     return df
        
    def summary(self):
        # print("Properties of the object:")
        keys = list(self.properties.keys())
        size = [None]*len(keys)
        for n,k in enumerate(keys):
            tmp = list(self.properties[k].shape[1:])
            if len(tmp) == 0 :
                size[n] = 1
            elif len(tmp) == 1:
                size[n] = tmp[0]
            else :
                size[n] = tmp
        df = pd.DataFrame(columns=["name","unit","shape"])
        df["name"] = keys
        df["unit"] = [ self.units[k] for k in keys ]
        df["shape"] = size
        return df