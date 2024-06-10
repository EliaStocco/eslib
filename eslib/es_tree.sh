#!/bin/bash

# Check if a folder and level are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <folder> <level>"
    exit 1
fi

FOLDER=$1
LEVEL=$2

# Use tree to display the structure and filter out __pycache__ and non-python files
tree -a -L "$LEVEL" "$FOLDER" | awk '
BEGIN {
    IGNORECASE = 1
}
/^[[:space:]]*$/ {
    next
}
/^[[:space:]]*└──/ || /^[[:space:]]*├──/ || /^[[:space:]]*│/ {
    if ($2 ~ /__pycache__/) {
        next
    }
    if ($2 ~ /\.py$/ || $0 ~ /[[:space:]]*$/) {
        print $0
    }
}
'
