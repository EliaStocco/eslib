
from ase import Atoms
from ase.calculators.calculator import Calculator
from eslib.classes.io import pickleIO
from typing import List, TypeVar, Any

T = TypeVar('T', bound='eslibModel')

class eslibModel(pickleIO,Calculator):
    """
    Class for models using the eslib framework.

    This class is designed to be a base class for models that use the eslib framework.
    It provides a summary method that prints information about the model.
    """

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
