from ase import Atoms
from typing import List
from eslib.classes.io import pickleIO
from typing import Type, TypeVar
T = TypeVar('T', bound='DipoleModel')

class DipoleModel(pickleIO):
    # @abstractmethod
    def get(self,traj:List[Atoms],**argv):
        raise NotImplementedError("this method should be overwritten.")
        pass

    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        return super().from_pickle(file_path)
    
    def summary(self, string="\t"):
        return
