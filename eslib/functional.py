import functools
import warnings
from typing import Any, Callable
from functools import wraps
import numpy as np
import multiprocessing as mp

def extend2NDarray(func_1d) -> Callable:
    @wraps(func_1d)
    def wrapper(x, *args, axis=-1, use_parallel=False, **kwargs):
        if use_parallel and kwargs:
            raise ValueError("**kwargs not supported with use_parallel=True")

        x = np.asarray(x)
        x_moved = np.moveaxis(x, axis, 0)
        shape_rest = x_moved.shape[1:]
        indices = list(np.ndindex(shape_rest))

        def worker(idx):
            x1d = x_moved[(slice(None),) + idx]
            return func_1d(x1d, *args)

        if use_parallel:
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(worker, indices)
        else:
            results = [func_1d(x_moved[(slice(None),) + idx], *args, **kwargs)
                       for idx in indices]

        # Handle output
        first_result = results[0]
        if isinstance(first_result, np.ndarray):
            result_shape = first_result.shape
            stacked = np.stack(results).reshape(shape_rest + result_shape)
            return np.moveaxis(stacked, -1, axis)
        else:
            return np.array(results, dtype=object).reshape(shape_rest)

    return wrapper

def custom_deprecated(reason: str = "", name: str = "deprecated", warning:Warning=DeprecationWarning) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A decorator to mark functions as deprecated. It will result in a warning being emitted 
    when the function is used.

    Args:
        reason (str): A message that describes the reason why the function is deprecated.

    Returns:
        function: A decorator that wraps the original function and emits a deprecation warning 
        when it is called.

    Example:
        @custom_deprecated("use 'new_function' instead")
        def old_function(x, y):
            return x + y
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"Call to {name} function {func.__name__}. {reason}"
            warnings.warn(message, category=warning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Create the unsafe decorator using functools.partial
def unsafe(func: Callable[..., Any]) -> Callable[..., Any]:
    return custom_deprecated("this method has not been debugged", "unsafe", UserWarning )(func)

# Create the unsafe decorator using functools.partial
def improvable(func: Callable[..., Any]) -> Callable[..., Any]:
    return custom_deprecated("this method can be improved", "improvable", UserWarning )(func)
