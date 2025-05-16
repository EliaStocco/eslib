import contextlib
import itertools
import os
# from scipy.ndimage import gaussian_filter1d, generic_filter
import re
import sys
from copy import copy
import numpy as np
from typing import Any, List, Union

#------------------#
class FakeList:
    """
    A fake list implementation.
    """
    def __init__(self, value: Any, length: int):
        self.value = value
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: Union[int, slice]) -> Union[Any, List[Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return [self.value] * ((stop - start + step - 1) // step)
        elif 0 <= index < self.length:
            return self.value
        else:
            raise IndexError("FakeList index out of range")
        
    @property
    def shape(self) -> tuple:
        return (self.length,*np.asarray(self.value).shape)

def invert_indices(indices):
    """
    Given a list of indices that map atoms_A to atoms_B,
    returns the reverted indices that would restore atoms_A from atoms_B.
    """
    inverted_indices = np.argsort(indices)
    return inverted_indices

def unique_elements(lst):
    """
    Returns:
    - unique_lst: list of unique elements (preserving order)
    - indices: list mapping each element in lst to its index in unique_lst
    - inverse_indices: list of lists, where each sublist contains indices in lst where that unique element appears
    """
    seen = {}
    unique_lst = []
    indices = []
    inverse_indices = []

    for i, item in enumerate(lst):
        if item not in seen:
            seen[item] = len(unique_lst)
            unique_lst.append(item)
            inverse_indices.append([i])
        else:
            inverse_indices[seen[item]].append(i)
        indices.append(seen[item])
        
    assert len(inverse_indices) == len(unique_lst), \
        f"Number of unique elements ({len(unique_lst)}) does not match the number of inverse indices ({len(inverse_indices)})"
    assert len(indices) == len(lst), \
        f"Number of indices ({len(indices)}) does not match the length of the original list ({len(lst)})"

    return unique_lst, indices, inverse_indices


#---------------------------------------#
def phonopy2atoms(data):
    from ase import Atoms
    supercell_points = data["points"]

    # Extract atomic positions and symbols
    symbols = [entry["symbol"] for entry in supercell_points]
    positions = [entry["coordinates"] for entry in supercell_points]

    # Extract cell parameters from the supercell lattice
    lattice_matrix = data["lattice"]

    # Create an ase.Atoms object
    return Atoms(symbols=symbols, scaled_positions=positions, cell=lattice_matrix, pbc=True)

# @np.vectorize(signature="'(i),(),()->()'")

# #---------------------------------------#
# def sigma_out_of_target(array, target, sigma):
#     """
#     Compute the 'a' and 'b' arrays by measuring how far the smoothed 'array' is
#     from the 'target' in terms of standard deviations ('sigma').

#     Parameters:
#     - array: The input array of data.
#     - target: The target array or value.
#     - sigma: The standard deviation for smoothing.

#     Returns:
#     - a: Array 'a' representing how far 'array' is from 'target' in terms of standard deviations.
#     - b: Array 'b' representing how far the smoothed 'array' is from 'target' in terms of standard deviations.
#     """
#     # Apply Gaussian smoothing to the input array
#     shape = array.shape
#     N = array.shape[1]

#     smooth = np.full(shape, np.nan)
#     for n in range(N):
#         smooth[:, n] = gaussian_filter1d(array[:, n], sigma[n], axis=0)

#     # Calculate the Euclidean distance between the smoothed array and the input array
#     delta = np.abs(smooth - array)

#     # Apply Gaussian smoothing to the delta values
#     std = np.full(shape, np.nan)
#     for n in range(N):
#         std[:, n] = generic_filter(
#             delta[:, n], lambda x: np.std(x), size=int(sigma[n]), mode="constant"
#         )
#         # gaussian_filter1d(delta[:,n], sigma[n], axis=0)

#     # Calculate 'a' and 'b' arrays representing how far 'array' and 'smooth' are from 'target' in terms of standard deviations
#     a = np.full(shape, np.nan)
#     b = np.full(shape, np.nan)
#     for n in range(N):
#         a[:, n] = (array[:, n] - target[n]) / std[:, n]
#         b[:, n] = (smooth[:, n] - target[n]) / std[:, n]

#     return a, b

#---------------------------------------#
def find_files_by_pattern(folder, pattern, expected_count=None, file_extension=None):
    """
    Find files in a folder that match a specified pattern and optional file extension.

    Args:
        folder (str): The path to the folder to search in.
        pattern (str): The file pattern to match (e.g., "*.txt").
        expected_count (int, optional): The expected number of matching files.
            If provided, an error will be raised if the count doesn't match.
        file_extension (str, optional): The file extension to restrict the search to.

    Returns:
        list: A list of matching file paths.

    Raises:
        ValueError: If the expected_count is provided and doesn't match the actual count.
    """
    files = os.listdir(folder)
    matching_files = [None] * len(files)
    n = 0
    for filename in files:
        if pattern in filename:
            if file_extension is None or filename.endswith(file_extension):
                matching_files[n] = filename
                n += 1

    matching_files = matching_files[:n]

    if expected_count is not None and len(matching_files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} files, but found {len(matching_files)}."
        )

    for n, file in enumerate(matching_files):
        matching_files[n] = os.path.join(folder, file)

    if expected_count is not None and expected_count == 1:
        matching_files = matching_files[0]

    return matching_files

#---------------------------------------#
def remove_files_in_folder(folder, extension):
    # List all files in the folder
    files = os.listdir(folder)

    # Iterate over the files and delete files with the specified extension
    n = 0
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) and file.endswith(extension):
                os.remove(file_path)
                n += 1
            #     # print(f"Deleted: {file_path}")
            # else:
            #     # print(f"Skipped: {file_path} (not a file or wrong extension)")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
        print("removed {:d} files".format(n))
    return

#---------------------------------------#
def recursive_copy(source_dict: dict, target_dict: dict) -> dict:
    """
    Recursively copy keys and values from a source dictionary to a target dictionary, if they are not present in the target.

    This function takes two dictionaries, 'source_dict' and 'target_dict', and copies keys and values from 'source_dict' to 'target_dict'. If a key exists in both dictionaries and both values are dictionaries, the function recursively calls itself to copy nested keys and values. If a key does not exist in 'target_dict', it is added along with its corresponding value from 'source_dict'.

    Args:
        source_dict (dict): The source dictionary containing keys and values to be copied.
        target_dict (dict): The target dictionary to which keys and values are copied if missing.

    Returns:
        dict: The modified 'target_dict' with keys and values copied from 'source_dict'.

    Example:
        >>> dict_A = {"a": 1, "b": {"b1": 2, "b2": {"b2_1": 3}}, "c": 4}
        >>> dict_B = {"a": 10, "b": {"b1": 20, "b2": {"b2_2": 30}}, "d": 40}
        >>> result = recursive_copy(dict_A, dict_B)
        >>> print(result)
        {'a': 10, 'b': {'b1': 20, 'b2': {'b2_1': 3, 'b2_2': 30}}, 'd': 40}
    """
    for key, value in source_dict.items():
        if (
            isinstance(value, dict)
            and key in target_dict
            and isinstance(target_dict[key], dict)
        ):
            recursive_copy(value, target_dict[key])
        else:
            if key not in target_dict:
                target_dict[key] = value
    return target_dict

#---------------------------------------#
def add_default(dictionary: dict = None, default: dict = None) -> dict:
    """
    Add default key-value pairs to a dictionary if they are not present.

    This function takes two dictionaries: 'dictionary' and 'default'. It checks each key in the 'default' dictionary, and if the key is not already present in the 'dictionary', it is added along with its corresponding value from the 'default' dictionary. If 'dictionary' is not provided, an empty dictionary is used as the base.

    Args:
        dictionary (dict, optional): The input dictionary to which default values are added. If None, an empty dictionary is used. Default is None.
        default (dict): A dictionary containing the default key-value pairs to be added to 'dictionary'.

    Returns:
        dict: The modified 'dictionary' with default values added.

    Raises:
        ValueError: If 'dictionary' is not of type 'dict'.

    Example:
        >>> existing_dict = {'a': 1, 'b': 2}
        >>> default_values = {'b': 0, 'c': 3}
        >>> result = add_default(existing_dict, default_values)
        >>> print(result)
        {'a': 1, 'b': 2, 'c': 3}
    """
    if dictionary is None:
        dictionary = {}

    if not isinstance(dictionary, dict):
        raise ValueError("'dictionary' has to be of 'dict' type")

    return recursive_copy(source_dict=default, target_dict=dictionary)

# https://stackabuse.com/python-how-to-flatten-list-of-lists/
#---------------------------------------#
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

#---------------------------------------#
def get_all_system_permutations(atoms):
    species = np.unique(atoms)
    index = {key: list(np.where(atoms == key)[0]) for key in species}
    # permutations = {key: get_all_permutations(i) for i,key in zip(index.values(),species)}
    permutations = [get_all_permutations(i) for i in index.values()]
    return list(itertools.product(*permutations))

#---------------------------------------#
def get_all_permutations(v):
    tmp = itertools.permutations(list(v))
    return [list(i) for i in tmp]

#---------------------------------------#
def get_one_file_in_folder(folder, ext, pattern=None):
    files = list()
    for file in os.listdir(folder):
        if file.endswith(ext):
            if pattern is None:
                files.append(os.path.join(folder, file))
            elif pattern in file:
                files.append(os.path.join(folder, file))

    if len(files) == 0:
        raise ValueError("no '*{:s}' files found".format(ext))
    elif len(files) > 1:
        raise ValueError("more than one '*{:s}' file found".format(ext))
    return files[0]

#---------------------------------------#
def nparray2list_in_dict(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: nparray2list_in_dict(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [nparray2list_in_dict(item) for item in data]
    else:
        return data

#---------------------------------------#
# https://www.blog.pythonlibrary.org/2014/02/14/python-101-how-to-change-a-dict-into-a-class/
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    # ----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

#---------------------------------------#
def args_to_dict(args):
    tmp = copy(args)
    if isinstance(tmp, dict):
        return tmp
    else:
        return vars(tmp)

#---------------------------------------#
def get_attributes(obj):
    return [i for i in obj.__dict__.keys() if i[:1] != "_"]

#---------------------------------------#
def merge_attributes(A, B):
    attribs = get_attributes(B)
    for a in attribs:
        setattr(A, a, getattr(B, a))
    return A

#---------------------------------------#
def remove_empty_folder(folder_path, show=True):
    if is_folder_empty(folder_path):
        os.rmdir(folder_path)
        if show:
            print(f"Folder '{folder_path}' has been removed.")
        return True
    else:
        if show:
            print(f"Folder '{folder_path}' is not empty and cannot be removed.")
        return False

#---------------------------------------#
def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

#---------------------------------------#
@contextlib.contextmanager
def suppress_output(suppress=True):
    if suppress:
        with open(os.devnull, "w") as fnull:
            sys.stdout.flush()  # Flush the current stdout
            sys.stdout = fnull
            try:
                yield
            finally:
                sys.stdout = sys.__stdout__  # Restore the original stdout
    else:
        yield

#---------------------------------------#
def get_line_with_pattern(file, pattern):
    # Open the file and search for the line
    try:
        with open(file, "r") as f:
            for line in f:
                if pattern in line:
                    return line
            else:
                print("String '{:s}' not found in file '{:s}'".format(pattern, file))
    except FileNotFoundError:
        raise ValueError("File '{:s}' not found".format(file))
    except:
        raise ValueError("error in 'get_line_with_pattern'")
    
def check_pattern_in_file(file_path: str, pattern: str) -> bool:
    """
    Check if a specific pattern is present in a given file.

    Args:
        file_path (str): The path to the file to be checked.
        phrase (str): The phrase to search for within the file.

    Returns:
        bool: True if the phrase is found in the file, False otherwise.
    """
    with open(file_path, 'r') as file:
        for line in file:
            if pattern in line:
                return True
    return False

#---------------------------------------#
def extract_number_from_filename(filename: str) -> float:
    """
    Extract the first numerical value found in a filename.

    Args:
        filename (str): The filename from which to extract the numerical value.

    Returns:
        float: The numerical value extracted from the filename.
    """
    filename = os.path.basename(filename)
    # Use regular expressions to find the first numerical value in the filename
    match_obj = re.search(r'(\d+\.?\d*)', filename)
    if match_obj:
        return float(match_obj.group(1))
    return float('inf')  # If no number is found, return infinity so these files appear last

def extract_float(string:str)->np.ndarray:
    """
    Extract all the float from a string
    """
    elments = re.findall(r'[-+]?\d*\.\d+E[+-]?\d+', string)
    if elments is None or len(elments) == 0:
        raise ValueError("no float found")
    return np.asarray([float(a) for a in elments])


def get_file_size_human_readable(file_path):
    """
    Return disk allocation size of a file as (value, unit),
    where value is between 0 and 1024 and unit is one of B, KB, MB, GB, etc.
    """
    stat = os.stat(file_path)
    block_size = 512  # st_blocks is in 512-byte units (POSIX standard)
    size_bytes = stat.st_blocks * block_size

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024:
            return round(size, 2), unit
        size /= 1024.0
    return round(size, 2), units[-1]  # fallback to TB or higher
