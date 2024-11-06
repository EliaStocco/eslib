#!/bin/bash

# Specify the requirements file
REQUIREMENTS_FILE="${ESLIB}/requirements.txt"

# Check if the requirements file exists
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Flag to track missing packages
missing_packages=false

# Loop through each package in requirements.txt
while IFS= read -r package || [ -n "$package" ]; do
    # Strip any extra whitespace
    package=$(echo "$package" | xargs)
    
    # Check if the package is specified with a version
    if [[ "$package" == *"=="* || "$package" == *">="* || "$package" == *"<="* || "$package" == *">"* || "$package" == *"<"* ]]; then
        # Split package and version
        # echo "$package"
        # Extract the package name by removing everything after the first occurrence of one of the operators
        package=$(echo "$package" | sed 's/[<>=!].*//')
        # package_version=$(echo "$package" | sed 's/^[^<>=!]*//')
        # echo "$package_name"
    fi

    #     # Check if the specific version is installed
    #     installed_version=$(pip show "$package_name" | grep -i '^version:' | awk '{print $2}')
    #     if [[ "$installed_version" != "$package_version" ]]; then
    #         echo "Package '$package_name' is not installed or not at the required version ($package_version)."
    #         missing_packages=true
    #     fi
    # else
    
    # Check if the package is installed (any version)
    if ! pip show "$package" &> /dev/null; then
        echo "Package '$package' is not installed."
        missing_packages=true
    fi
    # fi
done < "$REQUIREMENTS_FILE"

# Final message
if $missing_packages; then
    echo "Some packages are missing or have incorrect versions. Please install them with:"
    echo "pip install -r $REQUIREMENTS_FILE"
else
    echo "All packages are installed and meet the required versions."
fi
