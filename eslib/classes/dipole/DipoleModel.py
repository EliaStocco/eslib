from ase import Atoms
from typing import List
from eslib.classes.io import pickleIO
from typing import Type, TypeVar
T = TypeVar('T', bound='DipoleModel')

class DipoleModel(pickleIO):

    # def compute(self,traj:List[Atoms],**argv):
    #     raise ValueError("This method should be overwritten.")

    # @abstractmethod
    def get(self,traj:List[Atoms],**argv):
        raise NotImplementedError("this method should be overwritten.")
        pass

    # @classmethod
    # def from_file(cls: Type[T], file_path: str) -> T:
    #     return super().from_pickle(file_path)
    
    def summary(self, string="\t"):
        print("\n{:s}Model type: {:s}".format(string,self.__class__.__name__))
        print("\tModel summary:")
        super(self).summary(string=string)
        
