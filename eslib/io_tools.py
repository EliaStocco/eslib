import json
import numpy as np
from typing import Any, Union
import logging
import os
import re
import subprocess
import glob
from eslib.functions import extract_number_from_filename
from io import TextIOWrapper
from ase.io import string2index
from eslib.classes.append import AppendableList

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

    # for n in range(5):
    #     try:
    #         ofile = f"{file}.{n}.json"
    #         with open(ofile, "w") as ff:
    #             json_string:str = json.dumps(data, cls=NumpyEncoder, indent=n)
    #             # json_string = json_string.replace('[\n        ', '[').replace(',\n        ', ', ').replace('\n    ]', ']')
    #             ff.write(json_string)
    #     except:
    #         pass
        
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

def read_Natoms_homogeneous(file_path:str):
    """
    Reads the first line of a (ext)xyz file to extracts the number of atoms.
    It will be assumed that all the snapshots have the same number of atoms
    
    Args:
        file_path (str): Path to the file from which to read.
        
    Returns:
        int: The integer found on the first line of the file.
        None: If no integer is found on the first line.
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()  # Read the first line and strip any surrounding whitespace
            # Search for an integer in the first line
            found = re.search(r'\d+', first_line)  # This will match the first sequence of digits
            if found:
                return int(found.group())
            else:
                print(f"No integer found in the first line of {file_path}")
                return None
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def count_lines(file_path):
    """
    Count the number of lines in a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The number of lines in the file.

    Raises:
        CalledProcessError: If the subprocess command fails.
    """
    # Use wc -l to count the number of lines in the file
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    line_count = int(result.stdout.split()[0])
    return line_count

# def count_lines(filename):
#     with open(filename, 'r') as file:
#         return sum(1 for line in file)
    

def count_lines(file_path):
    # Run the wc -l command and get the output
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    
    # The output will be in the format "lines filename"
    # We split by space and take the first part, which is the line count
    line_count = int(result.stdout.split()[0])
    
    return line_count

# # Example usage
# file_path = 'myfile.txt'
# lines = count_lines(file_path)
# print(f"The file {file_path} has {lines} lines.")

def add_index2file(file,N):
    dir_name = os.path.dirname(file)            # Extract the directory
    base_name, ext = os.path.splitext(os.path.basename(file))  # Extract the base name and extension
    new_output = os.path.join(dir_name, f"{base_name}.n={N}{ext}")  # Construct the new file path
    return new_output

def pattern2sorted_files(patter):
    matched_files = glob.glob(patter)
            
    # Raise an error if no files were found
    if not matched_files:
        raise ValueError("No files found")
    
    if len(matched_files) > 1:            
        try:
            matched_files = [ matched_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in matched_files ])) ]
        except:
            pass
    return matched_files

def load_data(file: str, **kwargs) -> np.ndarray:
    if file.endswith(".txt"):
        data = np.loadtxt(fname=file, **kwargs)
    elif file.endswith(".npy"):
        data = np.load(file, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file}")
    return data

def pattern2data(pattern:str)->np.ndarray:
    """
    Load data from multiple files matching a pattern.

    Parameters:
    pattern (str): A glob pattern to match files.

    Returns:
    np.ndarray: A 1D array of data, where each element is a 1D array loaded from a file matching the pattern.

    Raises:
    FileNotFoundError: If no files match the pattern.
    """
    all_files = pattern2sorted_files(pattern)
    if not all_files:
        raise FileNotFoundError(f"No files matched the pattern '{pattern}'")
    
    try:
        matched_files = [ matched_files[i] for i in np.argsort(np.asarray([ int(extract_number_from_filename(x)) for x in matched_files ])) ]
    except:
        pass
    
    results = [None]*len(all_files)
    for n,matched_file in enumerate(all_files):
        results[n] = load_data(matched_file)
    return np.asarray(results)

#------------------#
def is_convertible_to_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#------------------#
def integer_to_slice_string(index):
    """
    Convert integer index to slice string.

    Args:
        index: Index to convert.

    Returns:
        slice: Converted slice.
    """
    if isinstance(index, slice):
        return index
    
    if is_convertible_to_integer(index):
        index=int(index)

    if isinstance(index, int):
        return string2index(f"{index}:{index+1}")
    elif index is None:
        return slice(None,None,None)
    elif isinstance(index, str):
        try:
            return string2index(index)
        except:
            raise ValueError("error creating slice from string {:s}".format(index))
    # elif isinstance(index, slice):
    #     return index
    else:
        raise ValueError("`index` can be int, str, or slice, not {}".format(index))

#------------------#
def get_offset_ipi(file:TextIOWrapper,
               Nmax:int=1000000,
               line_offset_old:list=None,
               index:slice=None):
    """
    Get line offsets in a file.

    Args:
        file (TextIOWrapper): File object.
        Nmax (int, optional): Maximum number of offsets. Defaults to 1000000.
        line_offset_old (list, optional): Old line offsets. Defaults to None.
        index (slice, optional): Slice index. Defaults to None.

    Returns:
        list: List of line offsets.
    """
    # Read in the file once and build a list of line offsets
    n = 0 
    line_offset = [None]*Nmax
    if line_offset_old is not None:
        line_offset[:len(line_offset_old)] = line_offset_old
        n = len(line_offset_old)
        del line_offset_old
    offset = 0
    restart = False
    if n == 0 : file.seek(0) # start from the beginning of the file
    for line in file: # cycle over all lines
        if n >= Nmax: # create a bigger list
            restart = True
            break
        if line.replace(" ","").startswith("#"): # check if the line contains a comment
            line_offset[n] = offset 
            n += 1
        if index is not None and index.stop is not None and n >= index.stop: # stop
            break
        offset += len(line)        
    if restart: return get_offset_ipi(file, Nmax=Nmax * 2, line_offset_old=line_offset, index=index)
    file.seek(0)
    return line_offset[:n]

#------------------#
def read_comments_ipi(file:TextIOWrapper,
                      index:slice=None):
    """
    Read comments from an XYZ file.

    Args:
        file (TextIOWrapper): File object.
        index (slice, optional): Slice index. Defaults to None.

    Returns:
        list: List of comments.
    """
    offset = get_offset_ipi(file=file,index=index)
    if index is not None:
        offset = offset[index]
    comments = [None]*len(offset)
    for n,l in enumerate(offset):
        file.seek(l)
        comments[n] = file.readline()
    return comments

#------------------#
def get_comment_offsets_extxyz(file: TextIOWrapper, Nmax: int = 1000000, line_offset_old: list = None, index: slice = None):
    """
    Get line offsets for comment lines in an extxyz file.

    Args:
        file (TextIOWrapper): File object.
        Nmax (int, optional): Maximum number of offsets. Defaults to 1000000.
        line_offset_old (list, optional): Old line offsets. Defaults to None.
        index (slice, optional): Slice index. Defaults to None.

    Returns:
        list: List of line offsets for comment lines.
    """
    n = 0
    line_offset = [None] * Nmax
    if line_offset_old is not None:
        line_offset[:len(line_offset_old)] = line_offset_old
        n = len(line_offset_old)
        del line_offset_old
    offset = 0
    restart = False
    if n == 0:
        file.seek(0)  # start from the beginning of the file
    while True:
        if n >= Nmax:  # create a bigger list
            restart = True
            break
        line = file.readline()
        if not line:
            break
        num_atoms = int(line.strip())
        offset += len(line)
        comment_line = file.readline()
        if not comment_line:
            break
        line_offset[n] = offset
        n += 1
        offset += len(comment_line)
        for _ in range(num_atoms):
            line = file.readline()
            if not line:
                break
            offset += len(line)
        if index is not None and index.stop is not None and n >= index.stop:  # stop
            break
    if restart:
        return get_comment_offsets_extxyz(file, Nmax=Nmax * 2, line_offset_old=line_offset, index=index)
    file.seek(0)
    return line_offset[:n]

def read_comments_extxyz(file_path: str, index: slice = None):
    """
    Extract all comment lines from an extxyz file.

    Parameters:
    file_path (str): Path to the extxyz file.
    index (slice, optional): Slice index. Defaults to None.

    Returns:
    list: A list of comment lines.
    """
    with open(file_path, 'r') as file:
        offsets = get_comment_offsets_extxyz(file, index=index)
        if index is not None:
            offsets = offsets[index]
        comments = [None] * len(offsets)
        for n, offset in enumerate(offsets):
            file.seek(offset)
            comments[n] = file.readline().strip()
    return comments


#------------------#
def read_comments_extxyz(file_path):
    """
    Extract all comment lines from an extxyz file.

    Parameters:
    file_path (str): Path to the extxyz file.

    Returns:
    list: A list of comment lines.
    """
    with open(file_path, "r") as file:
        file.seek(0)
        comments = AppendableList()
        while True:
            line = file.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            comment_line = file.readline()
            comments.append(comment_line)
            
            for _ in range(num_atoms):
                line = file.readline()
                if not line:
                    raise ValueError("Unexpected end of file.")
                
    return comments.finalize()
            
            