#!/usr/bin/env python

"""
This script shows SLURM environment variables.

Usage:
    python slurm-info.py [-h] [-o OUTPUT]


Example:
    python slurm-info.py > slurm_info.txt
"""

import os
from datetime import datetime

def print_slurm_info():
    """
    Prints SLURM environment variables to the screen.

    Args:
        None

    Returns:
        None
    """
    # Retrieve the SLURM environment variables
    slurm_vars = {
        "Job ID"                : os.getenv("SLURM_JOB_ID"),
        "Job Name"              : os.getenv("SLURM_JOB_NAME"),
        "Node List"             : os.getenv("SLURM_JOB_NODELIST"),
        "Number of Nodes"       : os.getenv("SLURM_JOB_NUM_NODES"),
        "Partition"             : os.getenv("SLURM_JOB_PARTITION"),
        "Account"               : os.getenv("SLURM_JOB_ACCOUNT"),
        "Node ID"               : os.getenv("SLURM_NODEID"),
        "Total Number of Nodes" : os.getenv("SLURM_NNODES"),
        "Tasks Per Node"        : os.getenv("SLURM_TASKS_PER_NODE"),
        "Total Number of Tasks" : os.getenv("SLURM_NTASKS"),
        "CPUs Per Task"         : os.getenv("SLURM_CPUS_PER_TASK"),
        "Memory Per Node"       : os.getenv("SLURM_MEM_PER_NODE"),
        "Memory Per CPU"        : os.getenv("SLURM_MEM_PER_CPU"),
        "Memory Per Task"       : os.getenv("SLURM_MEM_PER_TASK"),
        "Submit Host"           : os.getenv("SLURM_SUBMIT_HOST"),
        "Submit Directory"      : os.getenv("SLURM_SUBMIT_DIR"),
        "Cluster Name"          : os.getenv("SLURM_CLUSTER_NAME"),
        "User"                  : os.getenv("SLURM_JOB_USER"),
        "Job Start Time"        : os.getenv("SLURM_JOB_START_TIME"),
        "Job End Time"          : os.getenv("SLURM_JOB_END_TIME"),
        "Date and Time"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    explanations = {
        "Job ID"                : "The unique identifier assigned to the job by the Slurm workload manager.",
        "Job Name"              : "The name assigned to the job.",
        "Node List"             : "The list of nodes allocated to the job.",
        "Number of Nodes"       : "The number of nodes allocated to the job.",
        "Partition"             : "The partition in which the job is running.",
        "Account"               : "The account name associated with the job.",
        "Node ID"               : "The ID of the current node within the job.",
        "Total Number of Nodes" : "Total number of nodes allocated to the job.",
        "Tasks Per Node"        : "Number of tasks to be initiated on each node.",
        "Total Number of Tasks" : "Total number of tasks in the job.",
        "CPUs Per Task"         : "Number of CPUs allocated per task.",
        "Memory Per Node"       : "Memory allocated per node.",
        "Memory Per CPU"        : "Memory allocated per CPU.",
        "Memory Per Task"       : "Memory allocated per task.",
        "Submit Host"           : "The hostname of the machine from which the job was submitted.",
        "Submit Directory"      : "The directory from which the job was submitted.",
        "Cluster Name"          : "The name of the cluster.",
        "User"                  : "The user who submitted the job.",
        "Job Start Time"        : "The start time of the job.",
        "Job End Time"          : "The expected end time of the job.",
        "Date and Time"         : "The current date and time."
    }

    # Print the SLURM variables to the screen
    print("\n\tSlurm environment variables:")
    key_length   = max([len(key) for key in slurm_vars.keys()])
    value_length = max([len(str(value)) for value in slurm_vars.values()])
    for key, value in slurm_vars.items():
        value = str(value)
        print(f"\t - {key:<{key_length}s} : {value:<{value_length}s}  | {explanations[key]}")
    print()

if __name__ == "__main__":
    print_slurm_info()

