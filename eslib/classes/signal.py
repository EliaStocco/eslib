from typing import TypeVar

import numpy as np

from eslib.classes.physical_tensor import PhysicalTensor

T = TypeVar('T', bound='Signal')
class Signal(PhysicalTensor):
    """Class to handle signals."""
    pass

    @classmethod
    def from_file(cls, **argv):
        obj = super().from_file(**argv)
        return cls(obj).to_data()

    def fluctuation(self:T,**argv):
        """Computes the fluctuation of the signal."""
        return np.var(self,**argv)