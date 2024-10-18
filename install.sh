#!/bin/bash

echo "Installing ESLIB ..."

# Determine the directory of the script
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Set ESLIB to the script directory
ESLIB="$SCRIPT_DIR"
export PATH="$PATH:${ESLIB}/eslib"
export PYTHONPATH="$PYTHONPATH:${ESLIB}/eslib"
export PATH="$PATH:${ESLIB}"
export PYTHONPATH="$PYTHONPATH:${ESLIB}"

# Add script subdirectories to PATH
for dir in "${ESLIB}"/eslib/scripts/* "${ESLIB}"/eslib/* ; do
    if [ -d "$dir" ]; then
        export PATH="$PATH:$dir"
        # Make Python files executable in each subdirectory
        find "$dir" -name "*.py" -type f -exec chmod +x {} \;
    fi
done

find "${ESLIB}/cluster/" -name "*.py" -type f -exec chmod +x {} \;
export PATH="${PATH}:${ESLIB}/cluster/"
export PYTHONPATH="${PYTHONPATH}:${ESLIB}/cluster/"
export PYTHONPATH="${PYTHONPATH}:${ESLIB}/eslib/eslib/fortran"

echo "Done. All the scripts should be available in the PATH."

source cluster/archive.sh