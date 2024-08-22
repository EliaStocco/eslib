#!/bin/bash

# Define the file where you want to save the job information
output_file="~/job_info.txt"

# Define the width of each column
col1_width=20
col2_width=60

# Capture the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# Gather SLURM job information
job_id=${SLURM_JOB_ID}
job_name=${SLURM_JOB_NAME}
working_dir=${SLURM_SUBMIT_DIR}

# Create a temporary file for the new job information
temp_file=$(mktemp)

# Write the new job information to the temporary file
{
  printf "%-${col1_width}s : %-${col2_width}s\n" "JOB_ID" "$job_id"
  printf "%-${col1_width}s : %-${col2_width}s\n" "JOB_NAME" "$job_name"
  printf "%-${col1_width}s : %-${col2_width}s\n" "WORKING_DIR" "$working_dir"
  printf "%-${col1_width}s : %-${col2_width}s\n" "START_TIME" "$start_time"
  
  # Your job commands go here
  # Example:
  # sleep 60

  # Capture the end time and calculate elapsed time
  end_time=$(date +"%Y-%m-%d %H:%M:%S")
  elapsed_time=$(date -d@"$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))" -u +%H:%M:%S)
  
  # Append end time and elapsed time to the temporary file
  printf "%-${col1_width}s : %-${col2_width}s\n" "END_TIME" "$end_time"
  printf "%-${col1_width}s : %-${col2_width}s\n" "ELAPSED_TIME" "$elapsed_time"

  # Print the dashed line at the end
  echo "----------------------------------------"
} > "$temp_file"

# Concatenate the new content with the existing content
if [ -f "$output_file" ]; then
  cat "$temp_file" "$output_file" > "${output_file}.tmp"
else
  mv "$temp_file" "$output_file"
fi

# Replace the old file with the updated one
mv "${output_file}.tmp" "$output_file"

# Clean up temporary files
rm -f "$temp_file"

