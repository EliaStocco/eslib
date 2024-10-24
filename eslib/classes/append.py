from typing import TypeVar, Union

import numpy as np

T = TypeVar('T',bound="AppendableArray")


class AppendableArray:
    """
    A class that can append numpy arrays to a pre-allocated array.
    If the array is full, it doubles its size.
    """

    def __init__(self:T, size: int = 100000) -> None:
        """
        Initialize a new AppendableArray.

        Args:
            size (int): The initial size of the array. Defaults to 100000.
        """
        self._arr: np.ndarray = np.full(size, np.nan)
        self._size: int = 0
        self._max_size: int = size
        self._n_update: int = 0

    def append(self:T, x: Union[np.ndarray,float]) -> None:
        """
        Append an array to the array.

        Args:
            x (Union[np.ndarray, float]): The array to append. It can be a single float or a numpy array.

        """
        # Check if x is a single float
        if np.ndim(x) == 0:
            # If the array is full, double its size
            if self._size + 1 > self._max_size:
                self._expand()
            # Append the float to the array
            self._arr[self._size] = x
            # Increment the size counter
            self._size += 1
        else:  # x is an array
            # If the array is not large enough to hold x, double its size
            if self._size + len(x) > self._max_size:
                self._expand()
            # Append the array to the array
            self._arr[self._size:self._size + len(x)] = x
            # Increment the size counter
            self._size += len(x)

    def _expand(self:T) -> None:
        """
        Double the size of the array.
        """
        new_size: int = self._max_size * 2
        new_arr: np.ndarray = np.full(new_size, np.nan)
        new_arr[:self._size] = self.finalize()
        self._arr = new_arr
        self._max_size = new_size
        self._n_update += 1

    def finalize(self:T) -> np.ndarray:
        """
        Return the final array.

        Returns:
            T: The final array.
        """
        return self._arr[:self._size]

    



    