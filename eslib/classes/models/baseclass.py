
import pickle
from typing import Any, List, Type, TypeVar
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator

from eslib.classes.io import pickleIO

T = TypeVar('T', bound='eslibModel')

class eslibModel(pickleIO,Calculator):
    """
    Class for models using the eslib framework.

    This class is designed to be a base class for models that use the eslib framework.
    It provides a summary method that prints information about the model.
    """
    
    @classmethod
    def from_pickle(cls: Type[T], file: str) -> T:
        """Load an object from a *.pickle file."""
        # overloaded method from `pickleIO`
        try:
            with open(file, 'rb') as ff:
                obj:T = pickle.load(ff)
            if hasattr(obj, '__post__from_pickle__'):
                obj.__post__from_pickle__()
            if not isinstance(obj, cls):
                warn(f"Invalid pickle file format. Expected type: {cls.__name__}, got {type(obj).__name__}")
            return obj
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            print(f"Error loading from pickle file: {e}")
            return obj

    def summary(self:T, string: str = "\t") -> None:
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

    # @abc.abstractmethod
    def compute(self:T, traj: List[Atoms], prefix: str = "", raw: bool = False, **kwargs) -> Any:
        """
        Compute properties for a trajectory.

        Args:
            traj (List[Atoms]): List of ASE Atoms objects representing the trajectory.
            prefix (str, optional): Prefix to add to property names. Defaults to "".
            raw (bool, optional): If True, return raw numpy arrays. Defaults to False.
            **kwargs: Additional arguments to pass to compute function.

        Returns:
            Any: Computed properties.
        """
        raise NotImplementedError("this method should be overwritten.")
