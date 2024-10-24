# This is a new class to unifiy Properties and AtomicStructures
import abc
from typing import TypeVar

import numpy as np
import pandas as pd

from eslib.classes.io import pickleIO

# TypeVar for the Trajectory class
T = TypeVar('T', bound='Trajectory')


class Trajectory(abc.ABC,pickleIO):
    """
    Abstract base class for trajectories.

    This class defines two abstract methods that have to be implemented by subclasses:
    - set: sets the values of a specified attribute for all structures.
    - get: gets the values of a specified attribute for all structures.
    """

    @abc.abstractmethod
    def set(self:T, name:str, data:np.ndarray, what:str="unknown") -> None:
        """
        Set the values of a specified attribute for all structures.

        Parameters:
        - name (str): Name of the attribute.
        - data (np.ndarray): Array of values to set.
        - what (str): 'info' or 'arrays', if known. Default is 'unknown'.
        """
        raise NotImplementedError("this method should be overwritten by a child class.")

    @abc.abstractmethod
    def get(self:T, name:str, default:np.ndarray=None, what:str="all") -> np.ndarray:
        """
        Get the values of a specified attribute for all structures.

        Parameters:
        - name (str): Name of the attribute.
        - default (np.ndarray): Default value if attribute is missing. Default is None.
        - what (str): 'info' or 'arrays', if known. Default is 'all'.

        Returns:
        - np.ndarray: Array of attribute values.
        """
        raise NotImplementedError("this method should be overwritten by a child class.")
    
    @abc.abstractmethod
    def summary(self:T)->pd.DataFrame:
        pass
