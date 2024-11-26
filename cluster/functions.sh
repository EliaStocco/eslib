#!/bin/bash

# Function to extract addresses from a given file and log them
extract_addresses() {
    # Input file passed as the first argument
    local ipi_input="$1"

    # Check if file exists
    if [[ ! -f "${ipi_input}" ]]; then
        echo "File ${ipi_input} not found!"
        return 1  # Exit if the file doesn't exist
    fi

    # Append the header to log.out
    echo "Addresses in ${ipi_input}:" >> log.out

    # Set the Internal Field Separator to newline to handle each line properly
    IFS=$'\n'

    # Loop through each line containing "<address>"
    for line in $(grep "<address>" "${ipi_input}"); do
        echo "${line}" >> log.out  # Append each line to log.out
    done

    # Unset IFS to restore default behavior
    unset IFS
}

# Example usage:
# Call the function with the filename as the argument
# extract_addresses "input.xml"
