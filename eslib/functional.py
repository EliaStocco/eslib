import functools
import warnings

def deprecated(reason):
    """
    A decorator to mark functions as deprecated. It will result in a warning being emitted 
    when the function is used.

    Args:
        reason (str): A message that describes the reason why the function is deprecated.

    Returns:
        function: A decorator that wraps the original function and emits a deprecation warning 
        when it is called.

    Example:
        @deprecated("use 'new_function' instead")
        def old_function(x, y):
            return x + y
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Call to deprecated function {func.__name__}. {reason}"
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator
