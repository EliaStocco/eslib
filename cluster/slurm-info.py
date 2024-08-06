#!/usr/bin/env python

"""
This script writes SLURM environment variables to a file.

Usage:
    python slurm-info.py [-h] [-o OUTPUT]

Options:
    -h --help                   Show this screen.
    -o OUTPUT, --output OUTPUT  Output file for SLURM information [default: slurm_info.txt]

Example:
    python slurm-info.py -o slurm_info.txt
"""

import os
import argparse
from datetime import datetime

def write_slurm_info(output_file):
    """
    Writes SLURM environment variables to a file.

    Args:
        output_file (str): The path to the output file.

    Returns:
        None
    """
    # Retrieve the SLURM environment variables
    slurm_vars = {
        "Job ID": os.getenv("SLURM_JOB_ID"),
        "Job Name": os.getenv("SLURM_JOB_NAME"),
        "Node List": os.getenv("SLURM_JOB_NODELIST"),
        "Number of Nodes": os.getenv("SLURM_JOB_NUM_NODES"),
        "Partition": os.getenv("SLURM_JOB_PARTITION"),
        "Account": os.getenv("SLURM_JOB_ACCOUNT"),
        "Node ID": os.getenv("SLURM_NODEID"),
        "Total Number of Nodes": os.getenv("SLURM_NNODES"),
        "Tasks Per Node": os.getenv("SLURM_TASKS_PER_NODE"),
        "Total Number of Tasks": os.getenv("SLURM_NTASKS"),
        "CPUs Per Task": os.getenv("SLURM_CPUS_PER_TASK"),
        "Memory Per Node": os.getenv("SLURM_MEM_PER_NODE"),
        "Memory Per CPU": os.getenv("SLURM_MEM_PER_CPU"),
        "Memory Per Task": os.getenv("SLURM_MEM_PER_TASK"),
        "Submit Host": os.getenv("SLURM_SUBMIT_HOST"),
        "Submit Directory": os.getenv("SLURM_SUBMIT_DIR"),
        "Cluster Name": os.getenv("SLURM_CLUSTER_NAME"),
        "User": os.getenv("SLURM_JOB_USER"),
        "Job Start Time": os.getenv("SLURM_JOB_START_TIME"),
        "Job End Time": os.getenv("SLURM_JOB_END_TIME"),
        "Date and Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write the SLURM variables to the output file
    key_length = max([len(key) for key in slurm_vars.keys()])
    with open(output_file, 'w') as f:
        for key, value in slurm_vars.items():
            f.write(f"{key:<{key_length}s} : {value} \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write SLURM environment variables to a file.")
    parser.add_argument('-o', '--output', type=str, help='output file for SLURM information (default: %(default)s)', default='slurm_info.txt')
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call the write_slurm_info function with the output file path
    write_slurm_info(args.output)

