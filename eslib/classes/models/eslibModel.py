import numpy as np
from ase import Atoms
from typing import List, Dict, Any
from eslib.classes.io import pickleIO

class eslibModel(pickleIO):
    """
    Class for models using the eslib framework.

    This class is designed to be a base class for models that use the eslib framework.
    It provides a summary method that prints information about the model.
    """

    def summary(self, string: str = "\t") -> None:
        """
        Print summary of the model.

        Args:
            string (str): String to prepend to each line of the summary. 
                          Defaults to "\t".

        Returns:
            None
        """
        # Print the type of the model
        print(f"\n{string}Model type: {self.__class__.__name__}")
        # Print the summary of the model
        print(f"{string}Model summary:")
