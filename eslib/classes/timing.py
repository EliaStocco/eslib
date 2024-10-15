from contextlib import ContextDecorator
import time
from typing import Any, Optional, Tuple

def func_default(elapsed_time: float) -> None:
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

class timing(ContextDecorator):
    """
    A class to measure the execution time of code blocks and functions.

    This class can be used both as a context manager and a function decorator
    to measure and print the elapsed time of the wrapped code.

    Parameters
    ----------
    enabled : bool, optional
        Flag to enable or disable timing. Default is True.

    Methods
    -------
    __enter__():
        Starts the timer if timing is enabled.

    __exit__(*exc: Optional[Tuple[Any]]) -> None:
        Stops the timer and prints the elapsed time if timing is enabled.

    """
    
    enabled    : bool
    func       : callable
    start_time : float

    def __init__(self, enabled: bool = True,func:callable=func_default) -> None:
        """
        Initializes the timing class with the given enabled flag.

        Parameters
        ----------
        enabled : bool, optional
            Flag to enable or disable timing. Default is True.
        """
        self.enabled = enabled
        self.func    = func

    def __enter__(self) -> 'timing':
        """
        Starts the timer when entering the context or before function execution
        if timing is enabled.

        Returns
        -------
        timing
            The instance of the timing class.
        """
        if self.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc: Optional[Tuple[Any]]) -> None:
        """
        Stops the timer when exiting the context or after function execution,
        and prints the elapsed time if timing is enabled.

        Parameters
        ----------
        *exc : Optional[Tuple[Any]]
            Optional exception information.
        """
        if self.enabled:
            end_time = time.perf_counter()
            elapsed_time = end_time - self.start_time
            self.func(elapsed_time)
            # print(f"Elapsed time: {elapsed_time:.6f} seconds")

class MyClass:
    @timing(enabled=False)  # Disable timing for this method
    def my_method(self):
        for i in range(1000000):
            pass

def main():
    # Example using the `timing` class as a method decorator
    instance = MyClass()
    instance.my_method()  # This will print the elapsed time for `my_method` if enabled=True

    # Example using the `timing` class as a context manager
    with timing(enabled=True):
        # Code block to be timed
        total = sum(i for i in range(1000000))
        print(f"Total: {total}")

if __name__ == "__main__":
    main()

