#!/usr/bin/env python
import os
import re
import subprocess
from datetime import datetime

import colorama
import pandas as pd
from colorama import Fore, Style


colorama.init(autoreset=True)

# Use the login username from the environment rather than hard-coding it.
username = os.environ['USER']

# Get the list of SLURM job IDs for your user
job_list_command = f'squeue -u {username} -h -o "%A"'
job_list_output = subprocess.check_output(job_list_command, shell=True, universal_newlines=True)
# Split the job IDs into a list
job_ids = job_list_output.strip().split('\n')

# Initialize lists to store job information
job_names = []
job_ids_list = []
run_times = []
max_times = []
work_dirs = []
dependencies = []
num_nodes_list = []
job_states = []
start_times = []
wait_times = []


def format_duration(seconds):
    """Format a duration as a compact, human-readable value."""
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_start_time(value, now):
    """Return Slurm's predicted start timestamp and its offset from now."""
    if value in (None, "Unknown", "N/A"):
        return "Unknown", "Unknown"
    try:
        start = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return value, "Unknown"
    offset = (start - now).total_seconds()
    display = start.strftime("%Y-%m-%d %H:%M:%S")
    if offset < 0:
        return display, f"started {format_duration(offset)} ago"
    return display, format_duration(offset)


now = datetime.now()

# Process each job ID
for job_id in job_ids:
    if job_id == "":
        continue
    # Get job info for each job
    job_info_command = f'scontrol show job {job_id}'
    job_info_output = subprocess.check_output(job_info_command, shell=True, universal_newlines=True)

    # Use regular expressions to extract the values
    work_dir_match   = re.search(r'WorkDir=(\S+)', job_info_output)
    run_time_match   = re.search(r'RunTime=(\S+)', job_info_output)
    job_name_match   = re.search(r'JobName=(\S+)', job_info_output)
    max_time_match   = re.search(r'TimeLimit=(\S+)', job_info_output)
    dependency_match = re.search(r'Dependency=(\S+)', job_info_output)
    num_nodes_match  = re.search(r'NumNodes=(\d+)', job_info_output)
    job_state_match  = re.search(r'JobState=(\S+)', job_info_output)
    start_time_match = re.search(r'StartTime=(\S+)', job_info_output)

    if work_dir_match:
        work_dir = os.path.normpath(work_dir_match.group(1))
    else:
        work_dir = "not found"

    if run_time_match:
        run_time = run_time_match.group(1)
    else:
        run_time = "not found"

    if job_name_match:
        job_name = job_name_match.group(1)
    else:
        job_name = "not found"
    
    if max_time_match:
        max_time = max_time_match.group(1)
    else:
        max_time = "not found"

    if dependency_match:
        dependency = dependency_match.group(1)
        dependency = dependency.split("(")[0].split(":")
        if len(dependency) == 2:
            dependency[0] = dependency[0].split("after")[1]
            dependency = "{:s} [{:s}]".format(dependency[1], dependency[0])
        else:
            dependency = ""
    else:
        dependency = ""

    if num_nodes_match:
        num_nodes = num_nodes_match.group(1)
    else:
        num_nodes = "not found"

    if job_state_match:
        job_state = job_state_match.group(1)
    else:
        job_state = "not found"

    start_time, wait_time = format_start_time(
        start_time_match.group(1) if start_time_match else None, now
    )

    # Append the extracted values to the lists
    job_names.append(job_name)
    job_ids_list.append(job_id)
    run_times.append(run_time)
    work_dirs.append(work_dir)
    max_times.append(max_time)
    dependencies.append(dependency)
    num_nodes_list.append(num_nodes)
    job_states.append(job_state)
    start_times.append(start_time)
    wait_times.append(wait_time)

# Create a Pandas DataFrame
data = {
    'job name': job_names,
    'job ID': job_ids_list,
    'run time': run_times,
    'max time': max_times,
    'working directory': work_dirs,
    'dependencies': dependencies,
    'nodes': num_nodes_list,
    'job state': job_states,
    'start date/time': start_times,
    'time to start': wait_times
}

df = pd.DataFrame(data)

# Sort the DataFrame by job name
df.sort_values(by=['working directory', 'job ID'], inplace=True)

def lines(N):
    string = "\t|" + "-"*(N-3) + "|"
    print(string)

def max_len_col(df, col):
    if len(df) == 0:
        return 5
    else:
        return max(max([len(str(i)) for i in df[col]]), len(col))


# Colors used for the complete job row.
QUEUED_COLOR = Fore.YELLOW + Style.BRIGHT
RUNNING_COLOR = Fore.GREEN + Style.BRIGHT
DEPENDENCY_COLOR = Fore.CYAN + Style.BRIGHT


def job_color(row):
    """Return the color that describes the job's current scheduling state."""
    if row['dependencies']:
        return DEPENDENCY_COLOR
    if row['job state'] == 'RUNNING':
        return RUNNING_COLOR
    if row['job state'] == 'PENDING':
        return QUEUED_COLOR
    return ''

# Columns displayed in the status table. The last two are Slurm predictions.
columns = [
    "job name", "job ID", "run time", "max time", "dependencies", "nodes",
    "start date/time", "time to start", "working directory",
]
max_lengths = {column: max_len_col(df, column) for column in columns}


def border(width):
    print("\t|" + "-" * (width - 3) + "|")


print("\n")
header = "\t| " + " | ".join(
    f"{column:<{max_lengths[column]}}" for column in columns
) + " |"
border(len(header))
print(header)
border(len(header))

for _, row in df.iterrows():
    cells = []
    for column in columns:
        value = str(row[column])
        cells.append(
            f"{value:^{max_lengths[column]}}" if column == "nodes"
            else f"{value:<{max_lengths[column]}}"
        )
    line = "\t| " + " | ".join(cells) + " |"
    print(job_color(row) + line + Style.RESET_ALL)

border(len(header))
print("\n")
