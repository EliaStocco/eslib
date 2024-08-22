#!/bin/bash

# Get the job ID
job_id=$SLURM_JOB_ID

# Check if the job ID is available
if [ -z "$job_id" ]; then
    echo "No job ID found."
else
    echo "#---------------------------------------#"
    echo "# sacct"
    echo
    echo "sacct output for job $job_id:"
    sacct -X -j $job_id --format=JobID,JobName,MaxRSS,Elapsed,TotalCPU,UserCPU,SystemCPU
    echo

    echo "#---------------------------------------#"
    echo "# reportseff (if available)"
    echo

    # Check if the conda environment exists and activate it
    if conda info --envs | grep -q "seff"; then
        source activate seff
        reportseff ${job_id} --format "+jobname,start"
    else
        echo "The 'seff' conda environment is not available."
    fi
fi


