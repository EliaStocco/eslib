import json
import numpy as np
from typing import Any, Union
import logging
import os

#---------------------------------------#

def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Set up a logger that writes logs to a file and handles exceptions cleanly.

    Args:
        log_file (str): Path to the log file. If None, logs are not written to a file.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if log_file:
        # File handler to write logs to a file
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter with date and time
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add handler to the logger
        logger.addHandler(file_handler)

    # Log the working directory as the first line
    logger.debug(f"Logger initialized. Working directory: {os.getcwd()}")

    # Log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.warning("KeyboardInterrupt caught. Exiting.")
            exit(1)
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    import sys
    sys.excepthook = handle_exception

    return logger

# ---------------------------------------#
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy arrays.

    This encoder converts NumPy arrays into Python lists, ensuring compatibility
    with the JSON serialization format.

    Example:
        >>> import numpy as np
        >>> data = {"array": np.array([1, 2, 3])}
        >>> json.dumps(data, cls=NumpyEncoder)
        '{"array": [1, 2, 3]}'
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        return super().default(obj)


def save2json(file: str, data: dict) -> None:
    """
    Save a dictionary to a JSON file, handling NumPy arrays automatically.

    Args:
        file (str): The path to the JSON file.
        data (dict): The dictionary to save. Can include NumPy arrays.

    Example:
        >>> import numpy as np
        >>> data = {"scores": np.array([95, 85, 75])}
        >>> save2json("data.json", data)
    """
    with open(file, "w") as ff:
        json.dump(data, ff, cls=NumpyEncoder, indent=4)


# ---------------------------------------#
def convert_lists_to_arrays(data: Any) -> Any:
    """
    Recursively convert lists in a data structure to NumPy arrays where possible.

    Args:
        data (Any): The input data, typically a dictionary or list.

    Returns:
        Any: The data with lists converted to NumPy arrays where applicable.

    Example:
        >>> data = {"scores": [95, 85, 75], "info": {"ages": [21, 22, 23]}}
        >>> convert_lists_to_arrays(data)
        {'scores': array([95, 85, 75]), 'info': {'ages': array([21, 22, 23])}}
    """
    if isinstance(data, list):
        try:
            return np.array(data)  # Attempt to convert the list to a NumPy array
        except ValueError:
            return data  # If conversion fails, keep it as a list
    elif isinstance(data, dict):
        return {key: convert_lists_to_arrays(value) for key, value in data.items()}
    elif isinstance(data, (tuple, set)):
        return type(data)(convert_lists_to_arrays(value) for value in data)
    return data  # Return other data types unchanged


def read_json(file: str) -> dict:
    """
    Load a JSON file into a dictionary, converting lists to NumPy arrays where possible.

    Args:
        file (str): The path to the JSON file.

    Returns:
        dict: The loaded data with lists converted to NumPy arrays.

    Example:
        >>> data = {"scores": [95, 85, 75]}
        >>> with open("data.json", "w") as f:
        ...     json.dump(data, f)
        >>> json2dict("data.json")
        {'scores': array([95, 85, 75])}
    """
    with open(file, "r") as f:
        data = json.load(f)
    return convert_lists_to_arrays(data)
