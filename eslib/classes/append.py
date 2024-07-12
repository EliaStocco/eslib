import numpy as np
from typing import TypeVar


T = TypeVar('T')


class AppendableArray:
    """
    A class that can append numpy arrays to a pre-allocated array.
    If the array is full, it doubles its size.
    """

    def __init__(self, size: int = 100000) -> None:
        """
        Initialize a new AppendableArray.

        Args:
            size (int): The initial size of the array. Defaults to 100000.
        """
        self._arr: np.ndarray = np.full(size, np.nan)
        self._size: int = 0
        self._max_size: int = size
        self._n_update: int = 0

    def append(self, x: T) -> None:
        """
        Append an array to the array.

        Args:
            x (T): The array to append.

        """
        if self._size + len(x) > self._max_size:
            self._expand()
        self._arr[self._size:self._size + len(x)] = x
        self._size += len(x)

    def _expand(self) -> None:
        """
        Double the size of the array.
        """
        new_size: int = self._max_size * 2
        new_arr: np.ndarray = np.full(new_size, np.nan)
        new_arr[:self._size] = self.finalize()
        self._arr = new_arr
        self._max_size = new_size
        self._n_update += 1

    def finalize(self) -> T:
        """
        Return the final array.

        Returns:
            T: The final array.
        """
        return self._arr[:self._size]

    



    