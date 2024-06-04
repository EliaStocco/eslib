import re
import numpy as np

def extract_float(string:str)->np.ndarray:
    """
    Extract all the float from a string
    """
    elments = re.findall(r'[-+]?\d*\.\d+E[+-]?\d+', string)
    if elments is None or len(elments) == 0:
        raise ValueError("no float found")
    return np.asarray([float(a) for a in elments])