import os
import time
import numpy as np
from filelock import FileLock
from typing import List, Dict, Optional
from ase.io import read
from eslib.classes.models.mace_model import MACEModel
from eslib.io_tools import save2json
from .tools import check_exit, Logger


class FileIOBatchedMACE:
    """
    Handles batched input/output file processing with MACEModel predictions.
    """

    def __init__(self, folders: List[str], model: str, log_file: Optional[str] = None) -> None:
        """
        Initialize the class with folder paths, model, and optional logging.

        Args:
            folders (List[str]): Directories to monitor for input/output files.
            model (str): Path to the pre-trained MACE model file.
            log_file (Optional[str]): Log file path; logs to stdout if None.
        """
        self.folders = folders
        self.model = MACEModel.from_file(model)
        self.logger = Logger(log_file)

    def run(self) -> None:
        """
        Monitor folders, process input files with MACEModel, and save results.

        - Waits for input files to appear in specified folders.
        - Processes files in batches using the model.
        - Saves results to output JSON files.
        """
        self.logger.info("Starting batch processing.")
        N = len(self.folders)
        atoms = [None] * N
        ready = [False] * N
        single_results = [None] * N
        ifiles = [f"{folder}/input.extxyz" for folder in self.folders]
        ofiles = [f"{folder}/output.json" for folder in self.folders]

        while True:
            check_exit(self.logger)  # Graceful exit check

            # Reset readiness flags
            self.logger.debug("Resetting readiness flags.")
            ready = [False] * N

            # Wait for all input files
            self.logger.info("Waiting for input files.")
            
            # TODO
            # Here I should:
            #  - load the `self.model.batch_size` oldest files
            #  - call the network for these structures
            #  - write the output files 
            #  - start the loop again            
            while not all(ready):
                for n, (folder, ifile) in enumerate(zip(self.folders, ifiles)):
                    if os.path.exists(folder) and os.path.exists(ifile):
                        ready[n] = True
                        self.logger.debug(f"Input file found: {ifile}")
                check_exit(self.logger)
            self.logger.info("All input files found.")

            # Read atomic structures and delete input files
            self.logger.debug("Reading input files and removing them.")
            for n, ifile in enumerate(ifiles):
                with FileLock(f"{ifile}.lock"):
                    atoms[n] = read(ifile, format="extxyz", index=0)
                    os.remove(ifile)

            check_exit(self.logger)

            # Compute predictions with the MACE model
            self.logger.info("Running model predictions.")
            start_time = time.time()
            results: Dict[str, np.ndarray] = self.model.compute(atoms, raw=True)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Model computation completed in {elapsed_time:.2f} seconds.")

            # Process and store results per folder
            self.logger.debug("Processing results.")
            for n, _ in enumerate(single_results):
                single_results[n] = {}
                for key in results.keys():
                    value = np.take(results[key], indices=n, axis=0)
                    # If the value array has more than one element, use it as is, else convert it to a float
                    single_results[n][key] = value if value.size > 1 else float(value)


            # Write results to output files
            self.logger.info("Writing output files.")
            for ofile, res in zip(ofiles, single_results):
                with FileLock(f"{ofile}.lock"):
                    save2json(ofile, res)
                    self.logger.debug(f"Results written: {ofile}")
