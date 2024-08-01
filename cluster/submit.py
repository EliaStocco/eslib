#!/usr/bin/env python
import argparse
import subprocess
import os

SCRIPT = "main.sh"

def submit_jobs(n, filepath, dependency_type, existing_job_id=None):
    """
    Submit Slurm jobs with optional dependencies.

    Parameters:
        n (int): Number of jobs to submit.
        filepath (str): Filepath of the job script.
        dependency_type (str): Dependency type ('ok', 'any', or 'none').
        existing_job_id (int): Slurm job ID to use as a prerequisite.

    Returns:
        None
    """

    if not os.path.exists(filepath):
        raise ValueError("'{:s}' does not exist.".format(filepath))

    # Initialize job_id with an optional existing job ID
    job_id = existing_job_id
    
    # Submit subsequent jobs with dependencies
    for i in range(n):
        if dependency_type == 'ok':
            dependency_option = '--dependency=afterok'
        elif dependency_type == 'any':
            dependency_option = '--dependency=afterany'
        elif dependency_type == 'none':
            dependency_option = ''
        else:
            print("Invalid dependency type:", dependency_type)
            return

        # Construct the Slurm sbatch command
        command = ['sbatch']
        if job_id is not None:
            command.extend([dependency_option + ':' + str(job_id), filepath])
        else:
            command.append(filepath)
        
        # Print the sbatch command to the screen
        print(" ".join(command))
        
        # Submit the job with or without dependency and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Extract the job ID from the captured output
        output_lines = result.stdout.splitlines()
        print(output_lines,"\n")
        if output_lines:
            last_line = output_lines[-1]
            job_id_str = last_line.strip().split()[-1]
            try:
                job_id = int(job_id_str)
            except ValueError:
                print("Job ID is not an integer:", job_id_str)
        else:
            print("No output captured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Slurm jobs with dependencies")
    parser.add_argument("-n", type=int, metavar="\b",help="number of jobs to submit (default=%(default)s)", default=1)
    parser.add_argument("-f", type=str, metavar="\b",help="filepath of the job script (default=%(default)s)", default=SCRIPT)
    parser.add_argument("-d", type=str, metavar="\b",help="dependency type [ok, any, none] (default: %(default)s)", default="any",choices=["ok", "any", "none"])
    parser.add_argument("-e", type=int, metavar="\b",help="slurm job ID to use as a prerequisite (default: %(default)s)",default=None)
    args = parser.parse_args()

    submit_jobs(args.n, args.f, args.d, args.e)
