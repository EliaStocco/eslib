import os
import time
import shutil
import logging
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes
from dataclasses import dataclass, field
from typing import Dict, Union, Any
from eslib.io_tools import read_json

MANDATORY = ["energy","free_energy","forces","stress"]


def check_exit(logger: logging.Logger) -> None:
    """
    Checks for the existence of the 'EXIT' file. If found, terminates the program.

    Args:
        logger (logging.Logger): Logger to record the exit message.
    """
    if os.path.exists("EXIT"):
        logger.info("Exit signal detected. Terminating.")
        exit(0)


def prepare_folder(folder: str) -> None:
    """
    Prepares the folder by cleaning its contents or creating it if it doesn't exist.

    Args:
        folder (str): The path of the folder to prepare.
        logger (logging.Logger): Logger to record the actions.
    """
    if os.path.exists(folder):
        # Clean the folder by removing all its contents
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
    else:
        # Create the folder if it does not exist
        os.makedirs(folder)


@dataclass
class FileIOCalculator(Calculator):
    """
    A custom calculator that handles I/O operations with files and logs the process.

    Attributes:
        folder (str): Path to the folder where calculations are stored.
        log_file (str): Optional path to the log file (defaults to {folder}/log.out).
    """
    folder: str
    log_file: str = field(default=None)

    def __post_init__(self) -> None:
        """
        Initializes the calculator and prepares the folder for calculations.
        """
        super().__init__()
        os.makedirs(self.folder, exist_ok=True)

        # Default log file if not provided
        # if self.log_file is None:
        #     self.log_file = os.path.join(self.folder, "log.out")

        # Set up logging
        self.logger = self.setup_logging(self.log_file)
        self.logger.info("Initializing FileIOCalculator.")
        self.implemented_properties = MANDATORY

        # Prepare the folder
        prepare_folder(self.folder)

    def setup_logging(self, log_file: str) -> logging.Logger:
        """
        Set up the logger for the class.

        Args:
            log_file (str): Path to the log file.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file, mode='a') if log_file else logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def calculate(self, atoms: Atoms = None, properties=None, system_changes=all_changes) -> None:
        """
        Runs the calculation by writing input files, waiting for the output, and processing results.

        Args:
            atoms (Atoms): The atoms object containing atomic positions and configurations.
            properties: Additional properties required for the calculation (not used here).
            system_changes: Changes to the system (not used here).
        """
        start_time = time.time()  # Record the start time of the method
        self.logger.info("Starting calculation.")

        super().calculate(atoms, properties, system_changes)

        # Prepare the input file path
        ifile = os.path.join(self.folder, "input.extxyz")
        if os.path.exists(ifile):
            self.logger.error(f"Input file already exists: {ifile}")
            raise ValueError(f"Error: {ifile} should not exist yet.")

        # Write atomic structure to input file
        write(ifile, atoms, format="extxyz")
        self.logger.debug(f"Input file written: {ifile}")

        # Wait for output file to be created
        ofile = os.path.join(self.folder, "output.json")
        timeout_seconds = None  # Timeout after 10 minutes
        start_wait = time.time()

        while not os.path.exists(ofile):
            check_exit(self.logger)
            if timeout_seconds is not None:
                if time.time() - start_wait > timeout_seconds:
                    self.logger.error(f"Timeout waiting for output file: {ofile}")
                    raise TimeoutError(f"Output file not created within {timeout_seconds} seconds.")
            time.sleep(1)

        # Validate that the input file no longer exists
        if os.path.exists(ifile):
            self.logger.error(f"Input file still exists: {ifile}")
            raise ValueError(f"Error: {ifile} should not exist anymore.")

        # Read and validate the JSON output file
        data: Dict[str, Any] = read_json(ofile)
        self.logger.debug(f"Output file read: {ofile}")
        os.remove(ofile)

        # Process the results
        self.results: Dict[str, Union[float, np.ndarray, str]] = {}
        for key in MANDATORY:
            if key not in data:
                self.logger.warning(f"Missing key {key} in output data.")
                if key == "free_energy":
                    self.results["free_energy"] = data["energy"]
                elif key == "stress":
                    self.results["stress"] = np.zeros((3,3))
                else:
                    raise KeyError(
                        f"{key} is not in `data` (which was read from file '{ofile}'). "
                        f"`data` contains the following keys: {list(data.keys())}"
                    )
            else:
                self.results[key] = data[key]
                self.logger.debug(f"Processed key {key}")

        # Validate shapes of forces and stress
        got = self.results["forces"].shape
        exp = atoms.positions.shape
        if got != exp:
            self.logger.error(f"Forces have the wrong shape: got {got} but expected {exp}.")
            raise ValueError(f"Forces have the wrong shape: got {got} but expected {exp}.")

        got = self.results["stress"].shape
        exp = (3, 3)
        if got != exp:
            self.logger.error(f"Stress has the wrong shape: got {got} but expected {exp}.")
            raise ValueError(f"Stress has the wrong shape: got {got} but expected {exp}.")

        # Record end time and log the duration
        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Calculation completed successfully.")
        self.logger.info(f"Total time spent in calculate: {total_time:.2f} seconds.")

