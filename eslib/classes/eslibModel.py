import numpy as np
from ase import Atoms
from typing import List, Dict, Any
from eslib.classes.io import pickleIO

class eslibModel(pickleIO):
    
    def summary(self, string: str = "\t") -> None:
        print("\n{:s}Model type: {:s}".format(string,self.__class__.__name__))
        print("\tModel summary:")