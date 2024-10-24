#!/usr/bin/env python
import os
import re
import subprocess

import pandas as pd

# Replace 'your_username' with your actual SLURM username
username = 'elsto'

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

    # Append the extracted values to the lists
    job_names.append(job_name)
    job_ids_list.append(job_id)
    run_times.append(run_time)
    work_dirs.append(work_dir)
    max_times.append(max_time)
    dependencies.append(dependency)
    num_nodes_list.append(num_nodes)

# Create a Pandas DataFrame
data = {
    'job name': job_names,
    'job ID': job_ids_list,
    'run time': run_times,
    'max time': max_times,
    'working directory': work_dirs,
    'dependencies': dependencies,
    'nodes': num_nodes_list
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

# Calculate maximum string lengths for each column
max_lengths = {
    'job name': max_len_col(df, "job name"),
    'job ID': max_len_col(df, "job ID"),
    'run time': max_len_col(df, "run time"),
    'max time': max_len_col(df, "max time"),
    'dependency': max_len_col(df, "dependencies"),
    'working directory': max_len_col(df, "working directory"),
    'nodes': max_len_col(df, "nodes")
}

print("\n")

# Print the header
header = '\t| {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} |'.\
    format("job name", max_lengths['job name'], \
           "job ID", max_lengths['job ID'], \
           "run time", max_lengths['run time'], \
           "max time", max_lengths['max time'], \
           "dependency", max_lengths['dependency'], \
           "nodes", max_lengths['nodes'], \
           "working directory", max_lengths['working directory'])

lines(len(header))
print(header)
lines(len(header))

# Print the sorted DataFrame in the desired format
for index, row in df.iterrows():
    print('\t| {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:^{}} | {:<{}} |'.\
          format(row['job name'], max_lengths['job name'], row['job ID'], max_lengths['job ID'], row['run time'], max_lengths['run time'], row['max time'], max_lengths['max time'], row["dependencies"], max_lengths['dependency'], row['nodes'], max_lengths['nodes'], row['working directory'], max_lengths['working directory']))

lines(len(header))
print("\n")
