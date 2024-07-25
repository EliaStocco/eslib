#!/usr/bin/env python
import numpy as np
import xml.etree.ElementTree as ET
import ast
from typing import Tuple, List
from ase import Atoms
from ase.io import write
from eslib.tools import convert
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Convert an i-PI checkpoint file to an ASE readable file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"            , **argv,required=True , type=str     , help="input file")
    parser.add_argument("-o"  , "--output"           , **argv,required=True , type=str     , help="output file")
    parser.add_argument("-of" , "--output_format"    , **argv,required=False, type=str     , help="output file format (default: %(default)s)", default=None)
    return parser


#---------------------------------------#
def find_variable_in_xml(file_path: str, variable_name: str, dtype: type) -> Tuple[List, Tuple[int, ...]]:
    """
    Finds a variable in an XML file and returns its value and shape.

    The function parses the XML file, finds the specified variable, and
    extracts its value and shape. If the variable is not found, or if
    multiple instances of the variable are found, a ValueError is raised.

    Args:
        file_path (str): The path to the XML file.
        variable_name (str): The name of the variable to find.
        dtype (type): The data type of the variable.

    Returns:
        tuple: A tuple containing the value of the variable as a list of the
        specified data type and the shape of the variable.

    Raises:
        ValueError: If the variable is not found in the XML file or if multiple
        instances of the variable are found.

    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    elements = root.findall(f".//{variable_name}")

    if len(elements) == 0:
        raise ValueError(f"Could not find {variable_name} in {file_path}.")
    elif len(elements) > 1:
        raise ValueError(f"Found more than one {variable_name} in {file_path}.")

    element = elements[0]
    value = element.text
    value = value.replace("\n", "")  # Remove newlines
    value = value.replace(" ", "")  # Remove spaces
    value = value.replace("[", "")  # Remove square brackets
    value = value.replace("]", "")  # Remove square brackets
    value = [dtype(i) for i in value.split(",")]  # Split and convert to the desired data type
    shape = element.attrib.get('shape')  # Get the shape of the variable
    shape = ast.literal_eval(shape)  # Evaluate the shape as a tuple

    return value, shape


#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\n\tSearching for 'q' ... ", end="")
    q,sq = find_variable_in_xml(args.input, 'q', float)
    print("done")
    print("\tlen(q), shape:",len(q),",",sq)

    # print("\tSearching for 'p' ... ", end="")
    # p,sp = find_variable_in_xml(args.input, 'p', float)
    # print("done")
    # print("\tlen(p), shape:",len(p),",",sp)

    # print("\tSearching for 'm' ... ", end="")
    # m,sm = find_variable_in_xml(args.input, 'm', float)
    # print("done")
    # print("\tlen(m), shape:",len(m),",",sm)
    
    print("\n\tSearching for 'cell'  ... ", end="")
    c,cm = find_variable_in_xml(args.input, 'cell', float)
    print("done")
    print("\tlen(cell), shape:",len(c),",",cm)

    print("\n\tSearching for 'names' ... ", end="")
    n,sn = find_variable_in_xml(args.input, 'names', str)
    print("done")
    print("\tlen(names), shape:",len(n),",",sn)

    if sq[0] != 1:
        raise ValueError("q must have 1 bead only.")
    sq = sq[1]

    # if sp[0] != 1:
    #     raise ValueError("p must have 1 bead only.")
    # sp = sp[1]

    # if sq != sp:
    #     raise ValueError("q and p must have the same shape.")
    
    # if sm != sn:
    #     raise ValueError("m and names must have the same shape.")
    
    if cm != (3,3):
        raise ValueError("cell must have 3x3 shape.")
    
    if sq != 3*sn:
        raise ValueError("m must have 3*dof.")
    
    q = np.asarray(q).reshape(-1,3)
    # p = np.asarray(p).reshape(-1,3)
    # m = np.asarray(m)# .reshape(-1,3)
    c = np.asarray(c).reshape((3,3)).T

    print("\n\tConverting from bohr to angstrom from ... ", end="")
    q = convert(q,"length","atomic_unit","angstrom")
    c = convert(c,"length","atomic_unit","angstrom")
    print("done")
    
    print("\n\tCreating ase.Atoms object ... ", end="")
    atoms = Atoms(positions=q ,symbols=n,cell=c,pbc=True)
    print("done")
    # atoms.arrays['velocities'] = p/np.tile(m[:, np.newaxis], (1, 3))

    print("\n\tWriting to '{:s}' ... ".format(args.output), end="")
    write(args.output,atoms,format=args.output_format)
    print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()