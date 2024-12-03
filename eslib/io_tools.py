import json
import numpy as np
from typing import Any, Union

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
