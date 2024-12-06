import os
import numpy as np
import logging
from typing import List, Dict
from ase.io import read
from eslib.classes.models.mace_model import MACEModel
from eslib.io_tools import save2json

def check_exit() -> None:
    """
    Check if an 'EXIT' file exists in the current directory. 
    If it exists, terminate the program.
    """
    if os.path.exists("EXIT"):
        exit(0)


class FileIOBatchedMACE:
    """
    A class to run MACE model predictions in batch, processing multiple folders with input/output files.

    Attributes:
        folders (List[str]): List of folder paths to process.
        model (MACEModel): The MACE model object loaded from the specified file.
    """

    def __init__(self, folders: List[str], model: str, log_file: str = None) -> None:
        """
        Initializes the FileIOBatchedMACE class.
        
        Args:
            folders (List[str]): List of folder paths to process.
            model (str): Path to the model file.
            log_file (str, optional): Path to the log file. Defaults to None.
        """
        self.folders = folders
        self.model = MACEModel.from_file(model)
        
        # Set up logging
        self.logger = self.setup_logging(log_file)

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

    def run(self) -> None:
        """
        Run the batch processing of input/output files, including checking for files and 
        computing results using the MACE model. Writes results to output files.
        """
        N = len(self.folders)
        atoms = [None] * N
        ready = [False] * N
        single_results = [None] * N
        ifiles = [f"{folder}/input.extxyz" for folder in self.folders]
        ofiles = [f"{folder}/output.json" for folder in self.folders]

        while True:  # iterations
            check_exit()
            
            # Reset readiness status
            for n in range(N):
                ready[n] = False

            # Wait until all folders are ready
            while not all(ready):
                for n, (folder, ifile) in enumerate(zip(self.folders, ifiles)):
                    if os.path.exists(folder) and os.path.exists(ifile):
                        ready[n] = True
                check_exit()

            # Read atomic structures from input files
            for n, ifile in enumerate(ifiles):
                atoms[n] = read(ifile, format="extxyz", index=0)
                os.remove(ifile)

            check_exit()

            # Compute results using the MACE model
            self.logger.info("Running MACE model computation.")
            results: Dict[str, np.ndarray] = self.model.compute(atoms, raw=True)

            # Process and save results for each folder
            for n, _ in enumerate(single_results):
                single_results[n] = {}
                for key in results.keys():
                    value: np.ndarray = np.take(results[key], axis=0, indices=n)
                    single_results[n][key] = value if value.size > 1 else float(value)

            # Save results to output files
            for ofile, res in zip(ofiles, single_results):
                save2json(ofile, res)
                self.logger.info(f"Saved results to {ofile}.")
