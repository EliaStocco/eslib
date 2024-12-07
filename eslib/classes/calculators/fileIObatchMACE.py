import os
import numpy as np
from typing import List, Dict
from ase.io import read
from eslib.classes.models.mace_model import MACEModel
from eslib.io_tools import save2json
from .tools import check_exit, Logger

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
        self.logger = Logger(log_file)
    


    def run(self) -> None:
        """
        Run the batch processing of input/output files, including checking for files and 
        computing results using the MACE model. Writes results to output files.
        """
        self.logger.info("Starting calculation.")
        N = len(self.folders)
        atoms = [None] * N
        ready = [False] * N
        single_results = [None] * N
        ifiles = [f"{folder}/input.extxyz" for folder in self.folders]
        ofiles = [f"{folder}/output.json" for folder in self.folders]

        while True:  # iterations
            check_exit(self.logger)
            
            # Reset readiness status
            self.logger.debug("Resetting flags.")
            for n in range(N):
                ready[n] = False

            # Wait until all folders are ready
            self.logger.info("Waiting for input files.")
            while not all(ready):
                for n, (folder, ifile) in enumerate(zip(self.folders, ifiles)):
                    if os.path.exists(folder) and os.path.exists(ifile):
                        ready[n] = True
                        self.logger.debug(f"Input file {ifile} found.")
                check_exit(self.logger)
            self.logger.info("All input files found.")

            # Read atomic structures from input files
            self.logger.debug("Reading and removing input files.")
            for n, ifile in enumerate(ifiles):
                atoms[n] = read(ifile, format="extxyz", index=0)
                os.remove(ifile)

            check_exit(self.logger)

            # Compute results using the MACE model
            self.logger.info("Running MACE model.")
            results: Dict[str, np.ndarray] = self.model.compute(atoms, raw=True)

            # Process and save results for each folder
            for n, _ in enumerate(single_results):
                single_results[n] = {}
                for key in results.keys():
                    value: np.ndarray = np.take(results[key], axis=0, indices=n)
                    single_results[n][key] = value if value.size > 1 else float(value)

            # Save results to output files
            self.logger.info("Writing output files.")
            for ofile, res in zip(ofiles, single_results):
                save2json(ofile, res)
                self.logger.debug(f"Saved results to {ofile}.")
