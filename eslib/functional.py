import functools
import warnings
from typing import Any, Callable


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
