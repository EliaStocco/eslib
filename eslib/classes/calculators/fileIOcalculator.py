import os
import shutil
import json
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes
from dataclasses import dataclass
from typing import Dict, Union, Any
from eslib.io_tools import read_json

def check_exit():
    if os.path.exists("EXIT"):
        exit(0)

MANDATORY = ["energy", "free_energy", "forces", "stress"]

def prepare_folder(folder: str) -> None:
    if os.path.exists(folder):
        # Clean the folder by removing all its contents
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # Create the folder if it does not exist
        os.makedirs(folder)
    return

@dataclass
class FileIOCalculator(Calculator):
    folder: str

    def __post_init__(self):
        super().__init__()
        prepare_folder(self.folder)

    def calculate(self, atoms: Atoms = None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        
        # Prepare the input file path
        ifile = os.path.join(self.folder, "input.extxyz")
        if os.path.exists(ifile):
            raise ValueError(f"Error: {ifile} should not exist yet.")
        
        # Write atomic structure to input file
        write(ifile, atoms, format="extxyz")
        
        # Wait for output file to be created
        ofile = os.path.join(self.folder, "output.json")
        while not os.path.exists(ofile):
            check_exit()
        
        # Validate that the input file no longer exists
        if os.path.exists(ifile):
            raise ValueError(f"Error: {ifile} should not exist anymore.")
        
        # # Read and validate the JSON output file
        # with open(ofile, 'r') as file:
        #     data: Dict[str, Any] = json.load(file)
        data:Dict[str, Any] = read_json(ofile)
            
        # Process the results
        self.results:Dict[str, Union[float,np.ndarray,str]] = {}
        for key in MANDATORY:
            assert key in data, (
                f"{key} is not in `data` (which was read from file '{ofile}'). "
                f"`data` contains the following keys: {list(data.keys())}"
            )
            self.results = data[key]
            
        # self.results["energy"] = float(data["energy"])
        # self.results["free_energy"] = float(data["free_energy"])
        # self.results["forces"] = np.asarray(data["forces"])
        # self.results["stress"] = np.asarray(data["stress"])
        
        # Validate shapes of forces and stress
        got = self.results["forces"].shape
        exp = atoms.positions.shape
        assert got == exp, f"Forces have the wrong shape: got {got} but expected {exp}."
        
        got = self.results["stress"].shape
        exp = (3, 3)
        assert got == exp, f"Stress has the wrong shape: got {got} but expected {exp}."
        
        # Handle additional keys
        for key, value in data.items():
            if key in MANDATORY:
                continue
            try:
                # Try converting to float
                value = float(value)
            except (ValueError, TypeError):
                # Otherwise, try converting to numpy array
                try:
                    value = np.asarray(value)
                except Exception:
                    # Default to string if it can't be converted
                    value = str(value)
            self.results[key] = value
